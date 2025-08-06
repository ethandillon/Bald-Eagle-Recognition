"""
This script detects bald eagles in a YouTube livestream using a ResNet50 model.
It is designed to run continuously on a dedicated machine.
"""
# Core libraries
import os
import smtplib
from email.message import EmailMessage
from datetime import datetime
import tempfile
import time # For the main loop sleep
import random  # Add this import at the top of your file with the others
from datetime import timedelta


# Third-party libraries
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from dotenv import load_dotenv
import streamlink
from PIL import Image


error_email_sent_this_outage = False
# The script will run once per day at a random time between these hours.
RANDOM_WINDOW_START_HOUR = 8  # 8 AM
RANDOM_WINDOW_END_HOUR = 20 # 8 PM (20:00)
# --- Initial Setup ---
load_dotenv()
print("Loading ResNet50 model (this may take a moment)...")
model = ResNet50(weights='imagenet')
print("Model loaded successfully.")


# --- Helper Functions (send_email, smart_round) ---
def smart_round(number, sig_figs=2):
    if number == 0: return "0"
    else: return f"{number:.{sig_figs}g}"

def send_email(subject, content, recipient_email="ethanbradforddillon@gmail.com", sender_email="baldeagledetectionbot@gmail.com", image_path=None):
    password = os.getenv("EMAIL_PASSWORD")
    if not password:
        print("ERROR: Email password not found.")
        return False
    msg = EmailMessage()
    msg.set_content(content)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient_email
    if image_path:
        try:
            image = Image.open(image_path)
            img_type = image.format.lower()
            with open(image_path, 'rb') as f:
                img_data = f.read()
            msg.add_attachment(img_data, maintype='image', subtype=img_type, filename=os.path.basename(image_path))
        except Exception as e:
            print(f"ERROR: Could not attach image. Exception: {e}")
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, password)
            server.send_message(msg)
        print("Email sent successfully.")
        return True
    except Exception as e:
        print(f"ERROR: Failed to send email: {e}")
        return False

# --- Core Logic in a Reusable Function ---
def run_detection_cycle():
    """
    Runs a single cycle of the detection process with proper error handling.
    """
    global error_email_sent_this_outage

    YOUTUBE_URL = 'https://www.youtube.com/watch?v=B4-L2nfGcuE'
    temp_video_path = None

    try:
        # --- The ENTIRE success path is now inside this single 'try' block ---

        # 1. Get Stream and Capture Frame
        streams = streamlink.streams(YOUTUBE_URL)
        if not streams:
            raise ConnectionError("Streamlink could not find any available streams.")

        stream = streams['best']

        with tempfile.NamedTemporaryFile(suffix='.ts', delete=False) as temp_f:
            temp_video_path = temp_f.name
            with stream.open() as fd:
                chunk = fd.read(1024 * 1024 * 4)  # 4MB
                if not chunk:
                    raise ConnectionError("Read 0 bytes from streamlink. The stream might be dead.")
                temp_f.write(chunk)

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise ConnectionError(f"OpenCV could not open temp file: {temp_video_path}")

        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ConnectionError("OpenCV opened the temp file but could not read a frame from it.")

        print("Frame captured successfully.")

        # If we get here, the connection was successful. Reset the error flag if needed.
        if error_email_sent_this_outage:
            print("Connection successful. Resetting the error email notification flag.")
            error_email_sent_this_outage = False

        # 2. Prediction Logic
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        preds = model.predict(np.expand_dims(preprocess_input(frame_resized), axis=0))
        decoded_preds = decode_predictions(preds, top=10)[0]

        print("\n--- Top 10 Predictions ---")
        for i, (_, label, probability) in enumerate(decoded_preds, start=1):
            print(f"{i}. {label.replace('_', ' ').title()}: {probability:.1%}")
        print("--------------------------\n")

        # 3. Analyze and Act
        bald_eagle_is_detected = any(pred[1] == 'bald_eagle' for pred in decoded_preds)

        if bald_eagle_is_detected:
            eagle_pred = next(p for p in decoded_preds if p[1] == 'bald_eagle')
            eagle_probability = eagle_pred[2]

            print(f"SUCCESS: Bald Eagle detected with {eagle_probability:.1%} probability!")
            prob_text = f"Bald Eagle: {eagle_probability:.1%}"
            cv2.putText(frame, prob_text, (frame.shape[1] - 600, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"detection_{timestamp_str}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Image saved as {filename}")

            subject = "Bald Eagle Detection Alert!"
            content = f"A Bald Eagle was detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            send_email(subject, content, image_path=filename)
        else:
            print("Bald eagle not detected.")

    except Exception as e:
        # This 'except' block now catches any failure during the entire process.
        print(f"CRITICAL ERROR: An error occurred during the detection cycle. Details: {e}")

        if not error_email_sent_this_outage:
            print("Sending error notification email...")
            subject = "CRITICAL ERROR: Bald Eagle Bot - Cycle Failed"
            content = (
                f"The Bald Eagle detection script encountered a fatal error at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n\n"
                f"Error details:\n{e}"
            )
            send_email(subject, content)
            error_email_sent_this_outage = True
        else:
            print("An error email for this outage has already been sent. Suppressing new email.")

    finally:
        # This 'finally' block ensures the temporary file is ALWAYS cleaned up.
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)


def calculate_next_run_time():
    """
    Calculates a random runtime for the next available day within the defined window.
    """
    now = datetime.now()
    today_start_window = now.replace(hour=RANDOM_WINDOW_START_HOUR, minute=0, second=0, microsecond=0)

    # Determine if we should schedule for today or tomorrow
    if now < today_start_window:
        # It's before the window opens today, so schedule for today
        target_date = now.date()
    else:
        # It's either inside or after the window today, so schedule for tomorrow to be safe
        target_date = now.date() + timedelta(days=1)

    # Create the start and end datetime objects for the target date
    start_of_window = datetime(target_date.year, target_date.month, target_date.day, RANDOM_WINDOW_START_HOUR)
    end_of_window = datetime(target_date.year, target_date.month, target_date.day, RANDOM_WINDOW_END_HOUR)

    # Calculate the total duration of the window in seconds
    window_duration_seconds = (end_of_window - start_of_window).total_seconds()

    # Pick a random number of seconds to add to the start time
    random_seconds_offset = random.uniform(0, window_duration_seconds)

    # Calculate the final target time
    next_run_time = start_of_window + timedelta(seconds=random_seconds_offset)

    return next_run_time

# --- Main Application Loop ---
# --- Main Application Loop ---
def main():
    """
    The main entry point for the continuously running application.
    Schedules and runs the detection cycle once per day at a random time.
    """
    print("--- Bald Eagle Detection Bot Starting ---")
    print(f"Scheduling mode: Once per day between {RANDOM_WINDOW_START_HOUR}:00 and {RANDOM_WINDOW_END_HOUR}:00.")

    while True:
        # 1. Calculate the next scheduled run time
        next_run_time = calculate_next_run_time()
        print(f"\n--- Next detection cycle is scheduled for: {next_run_time.strftime('%Y-%m-%d %H:%M:%S')} ---")

        # 2. Calculate how long we need to sleep
        now = datetime.now()
        sleep_duration_seconds = (next_run_time - now).total_seconds()

        # 3. Sleep until the scheduled time (if it's in the future)
        if sleep_duration_seconds > 0:
            # We convert to hours/minutes for a more human-readable log message
            sleep_hours = sleep_duration_seconds / 3600
            print(f"--- Sleeping for {sleep_hours:.2f} hours... ---")
            time.sleep(sleep_duration_seconds)

        # 4. WAKE UP! It's time to run the detection.
        print(f"\n--- [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Waking up and starting detection cycle ---")
        run_detection_cycle()
        print("--- Cycle finished. Calculating next run time... ---")

        # The loop will now repeat, calculating a new time for the next day.


# --- Execution Block ---
if __name__ == "__main__":
    main()