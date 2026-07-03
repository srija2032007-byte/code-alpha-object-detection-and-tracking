import cv2
import time
import smtplib

def save_snapshot(frame):
    filename = f"alert_{int(time.time())}.jpg"
    cv2.imwrite(filename, frame)
    return filename


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

    except:
        print("Email alert failed (check credentials)")