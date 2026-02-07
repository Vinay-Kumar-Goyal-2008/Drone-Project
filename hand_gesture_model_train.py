import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

BATCH_SIZE = 10
EPOCHS = 70
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dirs = os.listdir("./gesture_data")

gesturedicts = {
    "swipe_left": 0,
    "swipe_right": 1,
    "swipe_up": 2,
    "swipe_down": 3,
    "open_palm": 4,
    "victory": 5,
    'flip':6,
    "None": 7
}

gesturedictsrev = {v: k for k, v in gesturedicts.items()}

# ---------------- LOAD DATA ----------------
x_list, y_list = [], []

for g in dirs:
    label = gesturedicts[g]
    for f in os.listdir(f"./gesture_data/{g}"):
        arr = np.load(f"./gesture_data/{g}/{f}")
        if arr.shape != (40, 21, 3):
            continue
        x_list.append(arr)
        y_list.append(label)

x = np.array(x_list)  # (N,40,21,3)
y = np.array(y_list)

xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, train_size=0.9, shuffle=True
)

# ---------------- DATASET ----------------
class GestureDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_ds = GestureDataset(xtrain, ytrain)
test_ds = GestureDataset(xtest, ytest)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- MODEL ----------------
class GestureNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

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

model = GestureNet(num_classes=len(gesturedictsrev)).to(DEVICE)

# ---------------- TRAINING ----------------
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

# ---------------- SAVE ----------------
torch.save(model.state_dict(), "model.pt")

with open("labels.pkl", "wb") as f:
    pickle.dump(gesturedictsrev, f)

print("Model saved as model.pt")
