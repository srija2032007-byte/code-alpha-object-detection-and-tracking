import cv2
import time
import smtplib
from ultralytics import YOLO

# =========================
# LOAD MODEL
# =========================
model = YOLO("yolov8n.pt")

# =========================
# SAVE SNAPSHOT
# =========================
def save_snapshot(frame):
    filename = f"alert_{int(time.time())}.jpg"
    cv2.imwrite(filename, frame)
    return filename


# =========================
# EMAIL ALERT
# =========================
def send_email_alert():
    try:
        sender = "your_email@gmail.com"
        password = "your_app_password"
        receiver = "receiver_email@gmail.com"

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, password)

        message = "ALERT: Person detected in surveillance system!"

        server.sendmail(sender, receiver, message)
        server.quit()

        print("Email sent successfully")

    except Exception as e:
        print("Email alert failed:", e)


# =========================
# CAMERA SETUP
# =========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not found")
    exit()

print("YOLO Detection Running... Press Q to exit")

alert_sent = False

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    results = model(frame)

    person_detected = False

    for result in results:
        boxes = result.boxes

        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":
                person_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, "PERSON", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # =========================
    # ALERT SYSTEM (ONCE ONLY)
    # =========================
    if person_detected and not alert_sent:
        print("Person detected! Sending alert...")
        snapshot = save_snapshot(frame)
        send_email_alert()
        alert_sent = True

    if not person_detected:
        alert_sent = False  # reset alert when person leaves

    # =========================
    # DISPLAY
    # =========================
    cv2.imshow("YOLO Surveillance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()