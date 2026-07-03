import cv2
import time
from ultralytics import YOLO

# =========================
# LOAD MODEL
# =========================
model = YOLO("yolov8n.pt")

# =========================
# CAMERA SETUP (Windows fix)
# =========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not found")
    exit()

# =========================
# VIDEO SAVE SETUP
# =========================
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

# =========================
# VARIABLES
# =========================
prev_time = 0

print("Level 2 AI System Running... Press Q to exit")

# =========================
# LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # =========================
    # YOLO TRACKING
    # =========================
    results = model.track(frame, persist=True)

    person_count = 0

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # COUNT PEOPLE ONLY
            if label == "person":
                person_count += 1

            # TRACK ID
            track_id = None
            if box.id is not None:
                track_id = int(box.id[0])

            # DRAW BOX
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text = f"{label} {conf:.2f}"
            if track_id is not None:
                text += f" ID:{track_id}"

            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

    # =========================
    # FPS CALCULATION
    # =========================
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # =========================
    # UI DISPLAY
    # =========================
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, f"People Count: {person_count}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # =========================
    # ALERT SYSTEM
    # =========================
    if person_count > 0:
        cv2.putText(frame, "PERSON DETECTED ⚠", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # =========================
    # SAVE VIDEO
    # =========================
    out.write(frame)

    # =========================
    # SHOW OUTPUT
    # =========================
    cv2.imshow("LEVEL 2 AI DETECTION SYSTEM", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video saved as output.mp4")