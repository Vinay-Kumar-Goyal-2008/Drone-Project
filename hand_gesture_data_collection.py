import cv2
import mediapipe as mp
import numpy as np
import os

gestures = [
    "swipe_left",
    "swipe_right",
    "swipe_up",
    "swipe_down",
    "open_palm",
    "victory",
    "flip",
    "None"
]

base_dir = "gesture_data"
frames_per_sample = 40
max_samples = 70

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

for g in gestures:
    os.makedirs(os.path.join(base_dir, g), exist_ok=True)

def next_index(gesture):
    return len(os.listdir(os.path.join(base_dir, gesture))) + 1

def rotation_normalize(joints):
    wrist = joints[0]
    index_mcp = joints[5]
    pinky_mcp = joints[17]

    x_axis = index_mcp - wrist
    x_axis /= (np.linalg.norm(x_axis) + 1e-8)

    temp = pinky_mcp - wrist
    z_axis = np.cross(x_axis, temp)
    z_axis /= (np.linalg.norm(z_axis) + 1e-8)

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= (np.linalg.norm(y_axis) + 1e-8)

    R = np.stack([x_axis, y_axis, z_axis], axis=0)
    return (R @ joints.T).T

current_gesture = None
recording = False
frame_count = 0
data = []

# ---------- FIXED KEY MAPPING (a–z) ----------
gesture_keys = {}
print("Select gesture using keys:")
for i, g in enumerate(gestures):
    key = chr(ord('a') + i)
    gesture_keys[key] = g
    print(f"{key} : {g}")
print("Press r to record, q to quit")
# --------------------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        if recording and frame_count < frames_per_sample:
            middlefinger = hand.landmark[9]
            wrist = hand.landmark[0]

            scale = (
                (middlefinger.x - wrist.x) ** 2 +
                (middlefinger.y - wrist.y) ** 2 +
                (middlefinger.z - wrist.z) ** 2
            ) ** 0.5 + 1e-8

            landmarks = []
            for lm in hand.landmark:
                landmarks.append([
                    (lm.x - wrist.x) / scale,
                    (lm.y - wrist.y) / scale,
                    (lm.z - wrist.z) / scale
                ])

            landmarks = np.array(landmarks, dtype=np.float32)
            landmarks = rotation_normalize(landmarks)

            data.append(landmarks)
            frame_count += 1

    if recording:
        cv2.putText(
            frame,
            f"{current_gesture} {frame_count}/{frames_per_sample}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        if frame_count == frames_per_sample:
            idx = next_index(current_gesture)
            path = f"{base_dir}/{current_gesture}/{current_gesture}_{idx}.npy"
            np.save(path, np.array(data))
            print("Saved", path)
            recording = False
            frame_count = 0
            data = []

    cv2.imshow("Gesture Recorder", frame)

    key = cv2.waitKey(1) & 0xFF
    key_char = chr(key)

    if key_char in gesture_keys:
        current_gesture = gesture_keys[key_char]
        print("Selected:", current_gesture)

    elif key == ord('r') and current_gesture:
        count = len(os.listdir(f"{base_dir}/{current_gesture}"))
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
