"""
This script detects bald eagles in a YouTube livestream using a ResNet50 model.
If an eagle is detected, it sends an email notification.
It is designed to be run locally for testing or deployed as an AWS Lambda function.
"""

# Core libraries
import os
import smtplib
from email.message import EmailMessage
from datetime import datetime

# Third-party libraries
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from dotenv import load_dotenv
import streamlink

# --- Initial Setup ---

# Load environment variables from a .env file for local development.
# This must be done before accessing os.getenv() for your password.
load_dotenv()

# Load the pre-trained ResNet50 model once.
# This is a heavy object, so loading it in the global scope is efficient,
# especially in a Lambda environment where it can be reused across invocations.
print("Loading ResNet50 model...")
model = ResNet50(weights='imagenet')
print("Model loaded successfully.")


# --- Helper Functions ---

def smart_round(number, sig_figs=2):
    """Rounds a number to a specified number of significant figures."""
    if number == 0:
        return "0"
    else:
        return f"{number:.{sig_figs}g}"


def send_email(subject, content, recipient_email="ethanbradforddillon@gmail.com",
               sender_email="baldeagledetectionbot@gmail.com"):
    """Sends an email using Gmail's SMTP server."""
    # The password is read from an environment variable for security.
    password = os.getenv("EMAIL_PASSWORD")
    if not password:
        print(
            "ERROR: Email password not found. Ensure it is set in your .env file or as a system environment variable.")
        return False

    msg = EmailMessage()
    msg.set_content(content)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient_email

    try:
        print("Connecting to email server...")
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, password)
            server.send_message(msg)
        print("Email sent successfully.")
        return True
    except Exception as e:
        print(f"ERROR: Failed to send email: {e}")
        return False


# --- Core Logic / Lambda Handler ---

def lambda_handler(event, context):
    """
    Main function to be executed. Connects to the stream, captures a frame,
    analyzes it, and sends an email if an eagle is detected.
    """
    YOUTUBE_URL = 'https://www.youtube.com/watch?v=B4-L2nfGcuE'

    # 1. Get Stream URL using streamlink
    try:
        print(f"Finding streams for {YOUTUBE_URL} using streamlink...")
        streams = streamlink.streams(YOUTUBE_URL)
        if not streams:
            print("ERROR: Could not find any streams on the provided URL.")
            return {"statusCode": 404, "body": "No streams found"}

        url = streams['best'].url
        print("Streamlink URL obtained successfully.")

    except Exception as e:
        print(f"ERROR: streamlink failed to get a stream URL. Exception: {e}")
        return {"statusCode": 500, "body": "Streamlink failed"}

    # 2. Capture a single frame from the stream
    cap = None
    frame = None
    try:
        # --- THE KEY FIX ---
        # Set an environment variable to increase FFmpeg's connection timeout.
        # This tells FFmpeg to wait up to 5 seconds to establish a connection,
        # which is crucial for slow-starting HLS streams.
        # The value is in microseconds. 5,000,000 us = 5 seconds.
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'timeout;5000000'
        # --- END OF FIX ---

        # We go back to NOT specifying the backend, letting OpenCV auto-detect.
        print("Opening video stream with increased timeout...")
        cap = cv2.VideoCapture(url)

        if not cap.isOpened():
            print("ERROR: Cannot open video stream even with increased timeout.")
            return {"statusCode": 500, "body": "Cannot open video stream"}

        # The retry loop is still good practice in case the first read is empty.
        max_retries = 5
        for attempt in range(max_retries):
            print(f"Attempting to read frame (Attempt {attempt + 1}/{max_retries})...")
            ret, frame = cap.read()
            if ret:
                print("Frame captured successfully!")
                break  # Exit the loop if we get a frame
            else:
                cv2.waitKey(500)  # Wait 0.5 seconds before the next try

        if frame is None:
            print(f"ERROR: Failed to capture frame after {max_retries} attempts.")
            return {"statusCode": 500, "body": "Failed to capture frame after retries"}

    except Exception as e:
        print(f"ERROR: An exception occurred during frame capture. Exception: {e}")
        return {"statusCode": 500, "body": "Exception during frame capture"}
    finally:
        if cap:
            cap.release()
            print("Video stream released.")

    # 3. Preprocess Frame and Make Prediction
    print("Preprocessing frame for model...")
    # ... (The rest of your code is perfect and does not need to change)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (224, 224))
    frame_expanded = np.expand_dims(frame_resized, axis=0)
    frame_preprocessed = preprocess_input(frame_expanded)

    print("Making prediction...")
    preds = model.predict(frame_preprocessed)
    decoded_preds = decode_predictions(preds, top=10)[0]

    detected_predictions_string = ""
    bald_eagle_is_detected = False
    for _, label, probability in decoded_preds:
        detected_predictions_string += f"Prediction: {label}, Probability: {smart_round(probability)}\n"
        if label == "bald_eagle":
            bald_eagle_is_detected = True

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if bald_eagle_is_detected:
        subject = "Bald Eagle Detection Alert!"
        content = f"A Bald Eagle was detected at {timestamp} on the livestream.\n\n--- Top Predictions ---\n{detected_predictions_string}"
        print("Bald eagle detected! Sending email notification.")
        send_email(subject, content)
    else:
        print("Bald eagle not detected. No email will be sent.")
        print(f"Top prediction was: {decoded_preds[0][1]}")

    return {
        "statusCode": 200,
        "body": f"Processing complete. Eagle detected: {bald_eagle_is_detected}"
    }

# --- Execution Block for Local Testing ---

# This block allows you to run the script directly from your computer.
# It will not be executed when deployed on AWS Lambda.
if __name__ == "__main__":
    print("--- Running a local test of the lambda_handler ---")
    lambda_handler(event=None, context=None)
    print("\n--- Local test finished ---")