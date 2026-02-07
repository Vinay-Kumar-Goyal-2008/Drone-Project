import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ===== MODEL PATH =====
MODEL_PATH = "efficientdet_lite0.tflite"

# ===== CREATE DETECTOR =====
BaseOptions = python.BaseOptions
ObjectDetectorOptions = vision.ObjectDetectorOptions
ObjectDetector = vision.ObjectDetector
VisionRunningMode = vision.RunningMode

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    score_threshold=0.5,
    running_mode=VisionRunningMode.IMAGE
)

detector = ObjectDetector.create_from_options(options)

# ===== CAMERA =====
cap = cv2.VideoCapture(0)

# ===== MAIN LOOP =====
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    for d in result.detections:
        print(d)
        bbox = d.bounding_box

        x1 = int(bbox.origin_x)
        y1 = int(bbox.origin_y)
        x2 = int(bbox.origin_x + bbox.width)
        y2 = int(bbox.origin_y + bbox.height)

        label = d.categories[0].category_name
        score = d.categories[0].score

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {score:.2f}",
            (x1, y1 - 10 if y1 > 20 else y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
