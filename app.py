import cv2
import yt_dlp
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import smtplib
from email.message import EmailMessage
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Load model once (outside handler to avoid reloading on every invocation)
model = ResNet50(weights='imagenet')

def smart_round(number, sig_figs=2):
    if number == 0:
        return "0"
    else:
        return f"{number:.{sig_figs}g}"

def send_email(subject, content, recipient_email="ethanbradforddillon@gmail.com", sender_email="baldeagledetectionbot@gmail.com"):
    password = os.getenv("EMAIL_PASSWORD")
    if not password:
        print("Email password not set. Set the EMAIL_PASSWORD environment variable.")
        return False

    msg = EmailMessage()
    msg.set_content(content)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient_email

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, password)
            server.send_message(msg)
        print("Email sent successfully")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

def lambda_handler(event, context):
    # Extract HLS URL
    ydl_opts = {'format': 'best'}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info('https://www.youtube.com/watch?v=B4-L2nfGcuE', download=False)
            url = info['url']
    except Exception as e:
        print(f"Failed to extract stream URL: {e}")
        return {"statusCode": 500, "body": "Failed to extract stream URL"}

    # Capture frame
    cap = cv2.VideoCapture(url)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to capture frame from the livestream.")
        return {"statusCode": 500, "body": "Failed to capture frame"}

    # Preprocess frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (224, 224))
    frame_resized = np.expand_dims(frame_resized, axis=0)
    frame_resized = preprocess_input(frame_resized)

    # Make prediction
    preds = model.predict(frame_resized)
    decoded_preds = decode_predictions(preds, top=10)[0]
    detected_predictions_string = ""
    bald_eagle_is_detected = False

    for decoded_pred in decoded_preds:
        probability = smart_round(decoded_pred[2], 2)
        detected_predictions_string += f"prediction: {decoded_pred[1]} probability: {probability}\n"
        if decoded_pred[1] == "bald_eagle":
            bald_eagle_is_detected = True

    # Send email based on detection
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if bald_eagle_is_detected:
        subject = "Bald Eagle Detection Alert"
        content = f"Bald Eagle Detected at {timestamp}!\n{detected_predictions_string}"
    else:
        subject = "Bald Eagle Not Detected"
        content = f"Bald Eagle Not Detected at {timestamp}!\n{detected_predictions_string}"

    send_email(subject, content)

    top_pred = decoded_preds[0]
    print(f"Top prediction: {top_pred[1]} with probability {top_pred[2]:.2f}")

    return {
        "statusCode": 200,
        "body": f"Processed frame. Top prediction: {top_pred[1]}"
    }