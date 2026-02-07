import cv2
import mediapipe as mp
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from time import time
import threading
import speech_recognition as sr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# -------------------- Zero-Shot Setup --------------------
candidate_labels = [
    "up", "down", " left", " right",
    "forward", "backward", "stop", "flip"
]
se = SentenceTransformer('all-MiniLM-L6-v2')
candidate_labels_encoded = se.encode(candidate_labels)

# -------------------- Config --------------------
FRAMES_PER_SAMPLE = 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESH = 0.85
VOTE_WINDOW = 5
MIN_VOTES = 3
COMBO_TIMEOUT = 1.5

speech_text = ""
speech_lock = threading.Lock()
stop_speech_thread = False

# -------------------- Gesture CNN --------------------
class GestureNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, (5,3), padding="same")
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2,1))
        self.conv2 = nn.Conv2d(32, 64, (3,3), padding="same")
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2,1))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64,128)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128,num_classes)

    def forward(self,x):
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
        return self.fc2(x)

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
        return self.fc2(x)

def load_model():
    with open("labels.pkl", "rb") as f:
        labels = pickle.load(f)
    with open("face_labels.pkl", "rb") as f:
        idx_to_label = pickle.load(f)

    model = GestureNet(len(labels))
    model.load_state_dict(torch.load("model.pt", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    face_model = FaceNet(len(idx_to_label))
    face_model.load_state_dict(torch.load("face_model.pt", map_location=DEVICE))
    face_model.to(DEVICE)
    face_model.eval()

    return model, labels, face_model, idx_to_label

# -------------------- MediaPipe --------------------
def init_mediapipe():
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    face_mesh = mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    return hands, mp_hands, mp.solutions.drawing_utils, face_mesh, mp_face

# -------------------- Hand Processing --------------------
def rotation_normalize(joints):
    wrist = joints[0]
    index_mcp = joints[5]
    pinky_mcp = joints[17]
    x_axis = index_mcp - wrist
    x_axis /= np.linalg.norm(x_axis)+1e-8
    temp = pinky_mcp - wrist
    z_axis = np.cross(x_axis,temp)
    if np.linalg.norm(z_axis)<1e-6:
        return joints
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis,x_axis)
    y_axis /= np.linalg.norm(y_axis)+1e-8
    R = np.stack([x_axis,y_axis,z_axis],axis=0)
    return (R @ joints.T).T

def extract_landmarks(hand):
    wrist = hand.landmark[0]
    middle = hand.landmark[9]
    scale = np.sqrt((middle.x-wrist.x)**2 + (middle.y-wrist.y)**2 + (middle.z-wrist.z)**2) + 1e-8
    joints = np.array([[(lm.x-wrist.x)/scale,(lm.y-wrist.y)/scale,(lm.z-wrist.z)/scale] for lm in hand.landmark])
    return rotation_normalize(joints)

def get_wrist_xy(hand):
    w = hand.landmark[0]
    return np.array([w.x,w.y])

# -------------------- Face Processing --------------------
def normalize_face(landmarks):
    landmarks = np.array(landmarks)
    nose = landmarks[1]
    landmarks -= nose
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    scale = np.linalg.norm(right_eye - left_eye) + 1e-8
    landmarks /= scale
    return landmarks

def extract_features(landmarks):
    indices = [33, 133, 468, 362, 263, 473]
    return landmarks[indices]

# -------------------- Hand Velocity --------------------
class HandVelocityTracker:
    def __init__(self, window=15):
        self.positions = deque(maxlen=window)
        self.times = deque(maxlen=window)
    def update(self,pos):
        self.positions.append(pos)
        self.times.append(time())
    def get_velocity(self):
        if len(self.positions)<2:
            return 0.0,0.0,0.0
        sum_vx=sum_vy=0.0
        valid=0
        for i in range(1,len(self.positions)):
            dx=self.positions[i][0]-self.positions[i-1][0]
            dy=self.positions[i][1]-self.positions[i-1][1]
            dt=self.times[i]-self.times[i-1]
            if dt<=1e-6:
                continue
            sum_vx+=dx/dt
            sum_vy+=dy/dt
            valid+=1
        if valid==0:
            return 0.0,0.0,0.0
        avg_vx=sum_vx/valid
        avg_vy=sum_vy/valid
        avg_speed = np.sqrt(avg_vx**2+avg_vy**2)
        return avg_vx,avg_vy,avg_speed

# -------------------- Gesture Prediction --------------------
def predict_gesture(model, buffer):
    x = torch.tensor(np.array(buffer),dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x),dim=1)[0]
    idx = torch.argmax(probs).item()
    return idx, probs[idx].item()

def stable_vote(pred_queue):
    classes = [p[0] for p in pred_queue]
    majority = max(set(classes), key=classes.count)
    confs = [c for cls,c in pred_queue if cls==majority]
    mean_conf = sum(confs)/len(confs)
    stable = classes.count(majority)>=MIN_VOTES and mean_conf>CONF_THRESH
    return stable, majority, mean_conf

def handle_combo(stable,majority,mean_conf,combo,labels):
    if not stable:
        return "",0.0
    now = time()
    combo.append((majority,now))
    while combo and now-combo[0][1]>COMBO_TIMEOUT:
        combo.popleft()
    if len(combo)>=2:
        g1,_ = combo[-2]
        g2,_ = combo[-1]
        if g1==4 and g2==5:
            combo.clear()
            return "OPEN PALM → VICTORY",1.0
    return labels[majority],mean_conf

# -------------------- Collision Avoidance --------------------
def get_collision_zone(frame):
    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 60, 150)
    zone_scores = np.zeros((3,3), dtype=np.int32)
    for y in range(h):
        for x in range(w):
            if edges[y, x] != 0:
                zy = min(y * 3 // h, 2)
                zx = min(x * 3 // w, 2)
                zone_scores[zy, zx] += 1
    min_pos = np.unravel_index(np.argmin(zone_scores), zone_scores.shape)
    zone_names = [
        ["Top-Left","Top-Center","Top-Right"],
        ["Middle-Left","Middle-Center","Middle-Right"],
        ["Bottom-Left","Bottom-Center","Bottom-Right"]
    ]
    return zone_names[min_pos[0]][min_pos[1]], zone_scores[min_pos]

# -------------------- Speech --------------------
def classify_command(text):
    maxsim=0
    index=0
    enc=se.encode(text)
    for i in range(len(candidate_labels_encoded)):
        sim=cosine_similarity(enc.reshape(1,-1),candidate_labels_encoded[i].reshape(1,-1))
        if maxsim<sim[0][0]:
            maxsim=sim[0][0]
            index=i
    return candidate_labels[index], maxsim

def extract_distance(text):
    match = re.search(r'(\d+(\.\d+)?)', text)
    return float(match.group(1)) if match else None

def parse_command(text):
    intent,conf = classify_command(text)
    distance = extract_distance(text)
    return intent,distance,conf

# -------------------- Speech Worker --------------------
def speech_worker():
    global speech_text, stop_speech_thread
    r = sr.Recognizer()
    r.dynamic_energy_threshold = True
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source,duration=1)
        while not stop_speech_thread:
            try:
                audio = r.listen(source)
                text = r.recognize_google(audio)
                with speech_lock:
                    speech_text = text.lower()
            except:
                pass

# -------------------- UI Overlay --------------------
def draw_overlay(frame, texts, font_scale=0.6, thickness=1, overlay_height=250):
    h,w,_ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay,(0,0),(w,overlay_height),(0,0,0),-1)
    cv2.addWeighted(overlay,0.6,frame,0.4,0,frame)
    y=25
    for text,color in texts:
        cv2.putText(frame,text,(10,y),cv2.FONT_HERSHEY_SIMPLEX,font_scale,color,thickness)
        y+=25

# -------------------- Main --------------------
def main():
    global stop_speech_thread
    model,labels,face_model,face_labels = load_model()
    hands, mp_hands, mp_draw, face_mesh, mp_face = init_mediapipe()
    threading.Thread(target=speech_worker,daemon=True).start()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    buffer = deque(maxlen=FRAMES_PER_SAMPLE)
    buffer_face = deque(maxlen=FRAMES_PER_SAMPLE)
    pred_queue = deque(maxlen=VOTE_WINDOW)
    combo = deque()
    velocity_tracker = HandVelocityTracker()
    default_move = "None"
    label = "None"

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)
        resultface = face_mesh.process(rgb)

        with speech_lock:
            speech = speech_text
            voice_intent, voice_distance, conf = parse_command(speech) if speech else ("None",0,0)

        if conf<0.2:
            voice_intent='None'
            voice_distance=0
        confd=(conf-0.2)/0.5

        vx=vy=speed=0.0
        pred_text="Waiting..."
        confidence=0.0
        avoidance_cmd = default_move
        zone_count=0

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            velocity_tracker.update(get_wrist_xy(hand))
            vx,vy,speed = velocity_tracker.get_velocity()
            buffer.append(extract_landmarks(hand))
            if len(buffer)==FRAMES_PER_SAMPLE:
                idx, conf_pred = predict_gesture(model, buffer)
                pred_queue.append((idx, conf_pred))
                if len(pred_queue)==VOTE_WINDOW:
                    stable, majority, mean_conf = stable_vote(pred_queue)
                    pred_text, confidence = handle_combo(stable, majority, mean_conf, combo, labels)
        else:
            if voice_intent=='None':
                avoidance_cmd, zone_count = get_collision_zone(frame)

        if resultface.multi_face_landmarks:
            lm=resultface.multi_face_landmarks[0]
            landmarks = np.array([[p.x, p.y, p.z] for p in lm.landmark])
            landmarks = normalize_face(landmarks)
            features = extract_features(landmarks)
            buffer_face.append(features)
            if len(buffer_face)==FRAMES_PER_SAMPLE:
                x=torch.tensor(np.array(buffer_face),dtype=torch.float32)
                x=x.permute(2,0,1).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    label=face_labels[face_model(x).argmax(1).item()]

        gesture_valid = pred_text not in ['None','', 'Waiting...'] and confidence>0.8
        voice_valid = voice_intent!='None' and conf>0.2

        if gesture_valid and (confidence>=conf or not voice_valid):
            final_cmd = pred_text
            final_dist = "Standard Allowed"
            final_conf = confidence
        elif voice_valid:
            final_cmd = voice_intent
            final_dist = voice_distance
            final_conf = confd
        else:
            final_cmd = avoidance_cmd
            final_dist = "Auto-Move"
            final_conf = 1.0

        texts = [
            (f"GESTURE: {pred_text} ({confidence:.2f})",(0,255,0)),
            (f"VELOCITY: vx={vx:.2f} vy={vy:.2f} speed={speed:.2f}",(255,255,0)),
            (f"SPEECH: {speech}",(255,0,255)),
            (f"VOICE INTENT: {voice_intent} | DIST: {voice_distance}",(0,255,255)),
            (f"FINAL COMMAND: {final_cmd} | DIST: {final_dist} | CONF: {final_conf:.2f}",(0,255,255)),
            (f"AVOIDANCE CMD: {avoidance_cmd} | Zone Count: {zone_count}",(0,128,255)),
            (f"Prediction: {label}",(0,255,255))
        ]

        draw_overlay(frame, texts)
        cv2.imshow("Drone Control: Gesture + Voice + Continuous Avoidance", frame)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break

    stop_speech_thread = True
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
