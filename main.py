import cv2
import yt_dlp
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import smtplib
from email.message import EmailMessage
import os
from datetime import datetime
import math

#Extract the HLS URL of the livestream
ydl_opts = {
    'format': 'best',  # Get the best available stream format
}
try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info('https://www.youtube.com/watch?v=B4-L2nfGcuE', download=False)
        url = info['url']  # HLS URL of the livestream
except Exception as e:
    print(f"Failed to extract stream URL: {e}")
    exit()

#Capture current frame from the livestream
cap = cv2.VideoCapture(url)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to capture frame from the livestream. Stream may be offline.")
    exit()

#Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Step 4: Preprocess the frame for the model
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_resized = cv2.resize(frame_rgb, (224, 224))
frame_resized = np.expand_dims(frame_resized, axis=0)
frame_resized = preprocess_input(frame_resized)

def smart_round(number, sig_figs=2):
    if number == 0:
        return "0"
    else:
        return f"{number:.{sig_figs}g}"


def bald_eagle_detected(detected_predictions_string):
    # Email configuration
    recipient_email = "ethanbradforddillon@gmail.com"
    sender_email = "baldeagledetectionbot@gmail.com"
    password = os.getenv("EMAIL_PASSWORD")

    if not password:
        print("Email password not set. Set the EMAIL_PASSWORD environment variable.")
        exit()

    # Prepare the email
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = EmailMessage()
    msg.set_content(f"Bald Eagle Detected!\n" + detected_predictions_string)
    msg['Subject'] = "Bald Eagle Detection Alert"
    msg['From'] = sender_email
    msg['To'] = recipient_email

    # Send the email
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, password)
            server.send_message(msg)
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")

def bald_eagle_not_detected(detected_predictions_string):
    # Email configuration
    recipient_email = "ethanbradforddillon@gmail.com"
    sender_email = "baldeagledetectionbot@gmail.com"
    password = os.getenv("EMAIL_PASSWORD")

    if not password:
        print("Email password not set. Set the EMAIL_PASSWORD environment variable.")
        exit()

    # Prepare the email
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = EmailMessage()
    msg.set_content(f"Bald Eagle Not Detected!\n" + detected_predictions_string)
    msg['Subject'] = "Bald Eagle Not Detected"
    msg['From'] = sender_email
    msg['To'] = recipient_email

    # Send the email
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, password)
            server.send_message(msg)
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")

#makes prediction based on frame
preds = model.predict(frame_resized)
#checks top 10 predictions
decoded_preds = decode_predictions(preds, top=10)[0]
detected_predictions_string = """"""
bald_eagle_is_detected = False
#for each of the top 10 predictions, adds to a string of the predictions and if a bald eagle is detected
#then sends an email with the list of top 10
for decoded_pred in decoded_preds:
    probability = smart_round(decoded_pred[2], 2)
    detected_predictions_string += f"prediction: {decoded_pred[1]} probability: {probability}\n"
    if decoded_pred[1] == "bald_eagle":
        bald_eagle_is_detected = True
if bald_eagle_is_detected:
    bald_eagle_detected(detected_predictions_string)
else:
    bald_eagle_not_detected(detected_predictions_string)

top_pred = decoded_preds[0]
print(f"Top prediction: {top_pred[1]} with probability {top_pred[2]:.2f}")


