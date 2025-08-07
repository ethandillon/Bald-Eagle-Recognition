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
import random
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

### ADDITION: Define how long to wait between retries after a connection failure.
RETRY_DELAY_MINUTES = 15 # Wait 15 minutes before retrying a failed cycle

# --- Initial Setup ---
load_dotenv()
print("Loading ResNet50 model (this may take a moment)...")
model = ResNet50(weights='imagenet')
print("Model loaded successfully.")


# --- Helper Functions (send_email, smart_round) ---
def smart_round(number, sig_figs=2):
    if number == 0: return "0"
    else: return f"{number:.{sig_figs}g}"

# ### MODIFICATION: The send_email function now accepts raw image data instead of a file path.
def send_email(subject, content, recipient_email="ethanbradforddillon@gmail.com", sender_email="baldeagledetectionbot@gmail.com", image_data=None, image_subtype='jpeg'):
    password = os.getenv("EMAIL_PASSWORD")
    if not password:
        print("ERROR: Email password not found.")
        return False
    msg = EmailMessage()
    msg.set_content(content)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient_email

    # Attach the image data if it was provided
    if image_data:
        try:
            # Generate a filename for the attachment
            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"detection_{timestamp_str}.{image_subtype}"
            # Add the attachment from the in-memory bytes
            msg.add_attachment(image_data, maintype='image', subtype=image_subtype, filename=filename)
        except Exception as e:
            print(f"ERROR: Could not attach image from memory. Exception: {e}")

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
    Returns True on success, False on failure.
    """
    global error_email_sent_this_outage

    YOUTUBE_URL = 'https://www.youtube.com/watch?v=B4-L2nfGcuE'
    temp_video_path = None

    try:
        # --- The ENTIRE success path is now inside this single 'try' block ---

        # 1. Get Stream and Capture Frame
        print("Attempting to connect to the stream...")
        streams = streamlink.streams(YOUTUBE_URL)
        if not streams:
            raise ConnectionError("Streamlink could not find any available streams. The stream might be offline or the URL is incorrect.")

        stream = streams['best']

        with tempfile.NamedTemporaryFile(suffix='.ts', delete=False) as temp_f:
            temp_video_path = temp_f.name
            with stream.open() as fd:
                chunk = fd.read(1024 * 1024 * 4)  # 4MB
                if not chunk:
                    raise ConnectionError("Read 0 bytes from streamlink. The stream might be dead or the connection was lost.")
                temp_f.write(chunk)

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise ConnectionError(f"OpenCV could not open temp file: {temp_video_path}")

        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ConnectionError("OpenCV opened the temp file but could not read a frame from it.")

        print("Frame captured successfully.")

        if error_email_sent_this_outage:
            print("Connection restored. Resetting the error email notification flag.")
            send_email("INFO: Bald Eagle Bot - Connection Restored",
                       f"The connection was successfully re-established at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
            error_email_sent_this_outage = False

        # 2. Prediction Logic
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        preds = model.predict(np.expand_dims(preprocess_input(frame_resized), axis=0))
        decoded_preds = decode_predictions(preds, top=5)[0]

        print("\n--- Top 5 Predictions ---")
        for i, (_, label, probability) in enumerate(decoded_preds, start=1):
            print(f"{i}. {label.replace('_', ' ').title()}: {probability:.1%}")
        print("-------------------------\n")

        # 3. Analyze and Act
        if not decoded_preds:
             print("Model returned no predictions.")
        else:
            top_pred_id, top_pred_label, top_pred_probability = decoded_preds[0]

            if top_pred_label == 'bald_eagle':
                print(f"SUCCESS: Bald Eagle is the #1 prediction with {top_pred_probability:.1%} probability!")

                # ### MODIFICATION: Draw the top 5 predictions onto the image frame
                y_pos = 70
                line_height = 50
                font_scale = 1.2
                font_thickness = 3

                # Add a semi-transparent background for readability
                cv2.rectangle(frame, (20, 20), (800, 30 + (len(decoded_preds) * line_height)), (0,0,0), -1)

                for i, (_, label, probability) in enumerate(decoded_preds, start=1):
                    text = f"{i}. {label.replace('_', ' ').title()}: {probability:.1%}"
                    # Add a white outline for the text
                    cv2.putText(frame, text, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness + 2)
                    # Add the main text color
                    cv2.putText(frame, text, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), font_thickness)
                    y_pos += line_height

                # ### MODIFICATION: Encode the image to memory instead of saving it to a file
                success, encoded_image = cv2.imencode('.jpg', frame)
                image_bytes = None
                if success:
                    image_bytes = encoded_image.tobytes()
                    print("Image encoded to memory successfully.")
                else:
                    print("ERROR: Failed to encode image to JPEG format.")

                # Send the email with the in-memory image data
                subject = "Bald Eagle Detection Alert! (#1 Prediction)"
                content = (
                    f"A Bald Eagle was detected as the #1 prediction at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n\n"
                    f"Confidence: {top_pred_probability:.1%}"
                )
                send_email(subject, content, image_data=image_bytes, image_subtype='jpeg')
            else:
                print(f"Bald eagle not the #1 prediction. Top result: {top_pred_label.replace('_', ' ').title()} ({top_pred_probability:.1%})")

        return True

    except Exception as e:
        print(f"CRITICAL ERROR: An error occurred during the detection cycle. Details: {e}")

        if not error_email_sent_this_outage:
            print("Sending error notification email...")
            subject = "CRITICAL ERROR: Bald Eagle Bot - Cycle Failed"
            content = (
                f"The Bald Eagle detection script encountered a fatal error at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n\n"
                f"This could be a temporary internet/stream issue. The script will automatically retry.\n\n"
                f"Error details:\n{e}"
            )
            send_email(subject, content)
            error_email_sent_this_outage = True
        else:
            print("An error email for this outage has already been sent. Suppressing new email.")

        return False

    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)


def calculate_next_run_time():
    """
    Calculates a random runtime for the next available day within the defined window.
    """
    now = datetime.now()
    today_start_window = now.replace(hour=RANDOM_WINDOW_START_HOUR, minute=0, second=0, microsecond=0)

    if now < today_start_window:
        target_date = now.date()
    else:
        target_date = now.date() + timedelta(days=1)

    start_of_window = datetime(target_date.year, target_date.month, target_date.day, RANDOM_WINDOW_START_HOUR)
    end_of_window = datetime(target_date.year, target_date.month, target_date.day, RANDOM_WINDOW_END_HOUR)
    window_duration_seconds = (end_of_window - start_of_window).total_seconds()
    random_seconds_offset = random.uniform(0, window_duration_seconds)
    next_run_time = start_of_window + timedelta(seconds=random_seconds_offset)
    return next_run_time


# --- Main Application Loop ---
def main():
    """
    The main entry point for the continuously running application.
    Schedules and runs the detection cycle once per day at a random time.
    Includes a retry loop for failed cycles.
    """
    print("--- Bald Eagle Detection Bot Starting ---")
    print(f"Scheduling mode: Once per day between {RANDOM_WINDOW_START_HOUR}:00 and {RANDOM_WINDOW_END_HOUR}:00.")
    print(f"Retry delay on failure: {RETRY_DELAY_MINUTES} minutes.")

    while True:
        next_run_time = calculate_next_run_time()
        print(f"\n--- Next detection cycle is scheduled for: {next_run_time.strftime('%Y-%m-%d %H:%M:%S')} ---")

        now = datetime.now()
        sleep_duration_seconds = (next_run_time - now).total_seconds()

        if sleep_duration_seconds > 0:
            sleep_hours = sleep_duration_seconds / 3600
            print(f"--- Sleeping for {sleep_hours:.2f} hours... ---")
            time.sleep(sleep_duration_seconds)

        print(f"\n--- [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Waking up to start detection task ---")

        while True:
            success = run_detection_cycle()
            if success:
                print("--- Cycle completed successfully. ---")
                break
            else:
                retry_seconds = RETRY_DELAY_MINUTES * 60
                print(f"--- Cycle failed. Retrying in {RETRY_DELAY_MINUTES} minutes... ---")
                time.sleep(retry_seconds)

        print("--- Task finished for this window. Calculating next run time... ---")


# --- Execution Block ---
if __name__ == "__main__":
    main()