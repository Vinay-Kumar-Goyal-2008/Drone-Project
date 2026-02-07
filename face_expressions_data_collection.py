import cv2
import mediapipe as mp
import numpy as np
import os

# ---------------- CONFIG ----------------
expressions = [
    "left_move",
    "right_move",
    "up_move",
    "down_move",
    "neutral"
]

base_dir = "face_data"
frames_per_sample = 40
max_samples = 50

mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# Create directories
for e in expressions:
    os.makedirs(os.path.join(base_dir, e), exist_ok=True)

# ---------------- HELPERS ----------------
def next_index(expression):
    return len(os.listdir(os.path.join(base_dir, expression))) + 1

def normalize_face(landmarks):
    landmarks = np.array(landmarks)
    nose = landmarks[1]
    landmarks -= nose
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    scale = np.linalg.norm(right_eye - left_eye) + 1e-8
    landmarks /= scale
    return landmarks

def extract_specific_features(landmarks):

    selected_indices = [
        33,
        133,
        468,
        362,
        263,
        473    
    ]

    return landmarks[selected_indices]

# ---------------- STATE ----------------
current_expression = None
recording = False
frame_count = 0
data = []

print("Select expression using number keys:")
for i, e in enumerate(expressions):
    print(i, e)
print("Press 'r' to record, 'q' to quit")

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        face = result.multi_face_landmarks[0]
        mp_draw.draw_landmarks(frame, face, mp_face.FACEMESH_TESSELATION)

        if recording and frame_count < frames_per_sample:
            landmarks = [[lm.x, lm.y, lm.z] for lm in face.landmark]
            landmarks = normalize_face(landmarks)               # Normalize for position and scale
            landmarks = extract_specific_features(landmarks)    # Keep only key expression features
            data.append(landmarks)

    if recording:
        cv2.putText(
            frame,
            f"{current_expression} {frame_count+1}/{frames_per_sample}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
        frame_count += 1

        if frame_count == frames_per_sample:
            data_np = np.array(data)
            idx = next_index(current_expression)
            path = f"{base_dir}/{current_expression}/{current_expression}_{idx}.npy"
            np.save(path, data_np)
            print("Saved", path)
            recording = False
            frame_count = 0
            data = []

    cv2.imshow("Face Expression Recorder", frame)
    key = cv2.waitKey(1) & 0xFF

    if key >= ord('0') and key <= ord(str(len(expressions) - 1)):
        current_expression = expressions[int(chr(key))]
        print("Selected:", current_expression)

    elif key == ord('r') and current_expression:
        count = len(os.listdir(f"{base_dir}/{current_expression}"))
        if count < max_samples:
            recording = True
            frame_count = 0
            data = []
            print("Recording...")
        else:
            print("Max samples reached")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
