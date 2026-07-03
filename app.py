import cv2
import streamlit as st
from ultralytics import YOLO
import time

model = YOLO("yolov8n.pt")

st.title("Object Detection & Tracking Web App")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

prev_time = 0

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not found")
        break

    frame = cv2.resize(frame, (640, 480))

    results = model.track(frame, persist=True)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

    FRAME_WINDOW.image(frame, channels="BGR")

cap.release()