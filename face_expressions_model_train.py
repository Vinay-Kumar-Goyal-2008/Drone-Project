import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ---------------- CONFIG ----------------
BATCH_SIZE = 8       # Face data is smaller, smaller batch is fine
EPOCHS = 70
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dirs = os.listdir("./face_data")  # Expression data folder

# Map expressions to labels
expr_dicts = {"left_move":0,"right_move":1,"up_move":2,"down_move":3,"neutral":4}
expr_dicts_rev = {v: k for k, v in expr_dicts.items()}

# ---------------- LOAD DATA ----------------
x_list, y_list = [], []

for e in dirs:
    print(e)
    label = expr_dicts[e]
    for f in os.listdir(f"./face_data/{e}"):
        arr = np.load(f"./face_data/{e}/{f}")
        if arr.shape[0] != 40:
            continue
        x_list.append(arr)
        y_list.append(label)

x = np.array(x_list)
y = np.array(y_list)

# Train-test split
xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, train_size=0.8, shuffle=True
)

# ---------------- DATASET ----------------
class FaceDataset(Dataset):
    def __init__(self, x, y):
        # Convert to torch tensor, permute to (batch, channels, frames, landmarks)
        self.x = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_ds = FaceDataset(xtrain, ytrain)
test_ds = FaceDataset(xtest, ytest)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- MODEL ----------------
class FaceNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Input shape: (batch, 3, frames, landmarks)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(5, 3), padding="same")
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 1))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding="same")
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 1))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, 128)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)

        x = torch.tanh(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x

model = FaceNet(num_classes=len(expr_dicts)).to(DEVICE)

# ---------------- TRAIN ----------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for xbatch, ybatch in train_loader:
        xbatch, ybatch = xbatch.to(DEVICE), ybatch.to(DEVICE)

        optimizer.zero_grad()
        logits = model(xbatch)
        loss = criterion(logits, ybatch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == ybatch).sum().item()
        total += ybatch.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss {total_loss:.4f} | Acc {acc:.4f}")

# ---------------- SAVE MODEL ----------------
torch.save(model.state_dict(), "face_model.pt")

with open("face_labels.pkl", "wb") as f:
    pickle.dump(expr_dicts_rev, f)

print("Face model saved as face_model.pt")
