import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from collections import deque

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAMES = 40

# ---------------- LOAD LABELS ----------------
with open("face_labels.pkl", "rb") as f:
    idx_to_label = pickle.load(f)

# ---------------- MODEL ARCH ----------------
class FaceNet(nn.Module):
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

# ---------------- LOAD MODEL ----------------
model = FaceNet(num_classes=len(idx_to_label)).to(DEVICE)
model.load_state_dict(torch.load("face_model.pt", map_location=DEVICE))
model.eval()

# ---------------- MEDIAPIPE ----------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------- NORMALIZATION ----------------
def normalize_face(landmarks):
    landmarks = np.array(landmarks)
    nose = landmarks[1]
    landmarks -= nose

    left_eye = landmarks[33]
    right_eye = landmarks[263]
    scale = np.linalg.norm(right_eye - left_eye) + 1e-8
    landmarks /= scale

    return landmarks

# ---------------- FEATURE SELECTION ----------------
def extract_features(landmarks):
    indices = [33, 133, 468, 362, 263, 473]  # eye landmarks only
    return landmarks[indices]

# ---------------- RUNTIME BUFFER ----------------
buffer = deque(maxlen=FRAMES)

# ---------------- CAMERA LOOP ----------------
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        lm = result.multi_face_landmarks[0].landmark
        landmarks = np.array([[p.x, p.y, p.z] for p in lm])

        landmarks = normalize_face(landmarks)
        features = extract_features(landmarks)
        buffer.append(features)

        if len(buffer) == FRAMES:
            x = np.array(buffer)
            x = torch.tensor(x, dtype=torch.float32)
            x = x.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(x)
                pred = logits.argmax(dim=1).item()

            label = idx_to_label[pred]

            cv2.putText(
                frame,
                f"Prediction: {label}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("Face Model Test", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
