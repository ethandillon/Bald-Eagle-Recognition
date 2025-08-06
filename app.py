"""
This script detects bald eagles in a YouTube livestream using a ResNet50 model.
If an eagle is detected, it saves an annotated image and sends an email notification with the image attached.
It uses modern libraries compatible with Python 3.11+.
"""
# Core libraries
import os
import smtplib
from email.message import EmailMessage
from datetime import datetime
import tempfile

# Third-party libraries
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from dotenv import load_dotenv
import streamlink
from PIL import Image # Modern library for image format detection

# --- Initial Setup ---
load_dotenv()
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

def send_email(subject, content, recipient_email="ethanbradforddillon@gmail.com", sender_email="baldeagledetectionbot@gmail.com", image_path=None):
    """Sends an email using Gmail's SMTP server, optionally with an image attachment."""
    password = os.getenv("EMAIL_PASSWORD")
    if not password:
        print("ERROR: Email password not found. Ensure it is set in your .env file or as a system environment variable.")
        return False

    msg = EmailMessage()
    msg.set_content(content)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient_email

    if image_path:
        try:
            # Use Pillow to get the image format
            image = Image.open(image_path)
            img_type = image.format.lower()

            with open(image_path, 'rb') as f:
                img_data = f.read()
            msg.add_attachment(img_data, maintype='image', subtype=img_type, filename=os.path.basename(image_path))
            print(f"Attached image: {os.path.basename(image_path)} (type: {img_type})")
        except Exception as e:
            print(f"ERROR: Could not attach image. Exception: {e}")

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
    This version uses a temporary file as a bridge for maximum reliability.
    """
    YOUTUBE_URL = 'https://www.youtube.com/watch?v=B4-L2nfGcuE'

    # 1. Get Stream Object using streamlink
    try:
        print(f"Finding streams for {YOUTUBE_URL} using streamlink...")
        streams = streamlink.streams(YOUTUBE_URL)
        if not streams:
            print("ERROR: Could not find any streams on the provided URL.")
            return {"statusCode": 404, "body": "No streams found"}

        # Get the stream object for the best available quality
        stream = streams['best']
        print("Stream object obtained successfully.")

    except Exception as e:
        print(f"ERROR: streamlink failed to get a stream object. Exception: {e}")
        return {"statusCode": 500, "body": "Streamlink failed"}

    # 2. Capture Frame using the Temporary File Bridge method
    temp_video_path = None
    frame = None
    try:
        # Create a temporary file that persists after closing.
        # The '.ts' suffix helps FFmpeg recognize the MPEG Transport Stream format.
        with tempfile.NamedTemporaryFile(suffix='.ts', delete=False) as temp_f:
            temp_video_path = temp_f.name
            print(f"Created temporary file: {temp_video_path}")

            # Read a chunk of the video stream and write it to the temp file.
            with stream.open() as fd:
                # 4MB should be plenty to contain at least one full video frame.
                chunk_size = 1024 * 1024 * 4
                print(f"Reading {chunk_size / 1024 / 1024:.1f}MB from stream into temp file...")
                chunk = fd.read(chunk_size)
                if not chunk:
                    raise IOError("Read 0 bytes from streamlink, the stream might be dead.")
                temp_f.write(chunk)
            print(f"Finished writing {len(chunk)} bytes to temp file.")

        # Use OpenCV to open the LOCAL temporary file, which is highly reliable.
        print(f"Opening local temp file with OpenCV...")
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise IOError(f"OpenCV could not open temp file: {temp_video_path}")

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise IOError("OpenCV opened the temp file but could not read a frame from it.")

        print("Frame captured successfully from temp file!")

    except Exception as e:
        print(f"ERROR: An exception occurred during frame capture. Exception: {e}")
        return {"statusCode": 500, "body": "Exception during frame capture"}
    finally:
        # CRITICAL: Always clean up the temporary file, even if errors occur.
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            print(f"Cleaned up temporary file: {temp_video_path}")

    # 3. Preprocess Frame and Make Prediction
    print("Preprocessing frame for model...")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (224, 224))
    frame_expanded = np.expand_dims(frame_resized, axis=0)
    frame_preprocessed = preprocess_input(frame_expanded)

    print("Making prediction...")
    preds = model.predict(frame_preprocessed)
    decoded_preds = decode_predictions(preds, top=10)[0]

    print("\n--- Top 10 Predictions ---")
    for i, (_, label, probability) in enumerate(decoded_preds, start=1):
        # We format the output for readability:
        # - Make the label title case (e.g., 'bald_eagle' -> 'Bald Eagle')
        # - Format the probability as a percentage with one decimal place
        human_readable_label = label.replace('_', ' ').title()
        print(f"{i}. {human_readable_label}: {probability:.1%}")
    print("--------------------------\n")

    # 4. Analyze Predictions, Annotate, and Act
    detected_predictions_string = ""
    bald_eagle_is_detected = False
    eagle_probability = 0.0

    for _, label, probability in decoded_preds:
        detected_predictions_string += f"Prediction: {label}, Probability: {smart_round(probability)}\n"
        if label == "bald_eagle":
            bald_eagle_is_detected = True
            eagle_probability = probability

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if bald_eagle_is_detected:
        # Annotate the original frame with the probability
        prob_text = f"Bald Eagle: {eagle_probability:.1%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_color = (0, 255, 255)  # Yellow in BGR format
        thickness = 3
        text_position = (frame.shape[1] - 600, 70)

        cv2.putText(frame, prob_text, text_position, font, font_scale, font_color, thickness, cv2.LINE_AA)

        # Save the annotated image with a unique filename
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"detection_{timestamp_str}.jpg"
        cv2.imwrite(filename, frame)
        print(f"SUCCESS: Eagle detected! Image saved as {filename}")

        # Send the email alert with the image attached
        subject = "Bald Eagle Detection Alert!"
        content = f"A Bald Eagle was detected at {timestamp} on the livestream.\n\n--- Top Predictions ---\n{detected_predictions_string}"
        send_email(subject, content, image_path=filename)
    else:
        print("Bald eagle not detected.")
        print(f"Top prediction was: {decoded_preds[0][1]}")

    return {
        "statusCode": 200,
        "body": f"Processing complete. Eagle detected: {bald_eagle_is_detected}"
    }


def test_email_functionality():
    """
    Runs a suite of tests to verify that the send_email function is working correctly.
    """
    print("\n--- Starting Email Functionality Test ---")

    # --- Test 1: Send a text-only email ---
    print("\n1. Testing text-only email...")
    text_subject = "Test Email: Text-Only"
    text_content = "This is a test of the text-only email sending functionality.\nIf you received this, it works!"

    # We can check the return value to see if it succeeded
    success = send_email(text_subject, text_content)
    if not success:
        print("   -> Text-only email test FAILED.")
        print("\n--- Email Functionality Test Finished ---")
        return  # No point in continuing if the basic test fails
    else:
        print("   -> Text-only email test SUCCEEDED (check your inbox).")

    # --- Test 2: Send an email with a dummy image attachment ---
    print("\n2. Testing email with image attachment...")
    dummy_image_path = "test_image.jpg"
    try:
        # Create a simple black dummy image using OpenCV
        print(f"   -> Creating a dummy image for testing: {dummy_image_path}")
        dummy_image = np.zeros((100, 200, 3), dtype=np.uint8)  # A 200x100 black rectangle
        cv2.imwrite(dummy_image_path, dummy_image)

        # Send the email with the attachment
        image_subject = "Test Email: With Image Attachment"
        image_content = "This is a test of the email sending functionality with an image attached.\nYou should see a small black rectangle attached to this email."

        success = send_email(image_subject, image_content, image_path=dummy_image_path)
        if not success:
            print("   -> Image attachment email test FAILED.")
        else:
            print("   -> Image attachment email test SUCCEEDED (check your inbox).")

    except Exception as e:
        print(f"   -> An error occurred during the image attachment test: {e}")
    finally:
        # CRITICAL: Clean up the dummy image file, no matter what happens
        if os.path.exists(dummy_image_path):
            os.remove(dummy_image_path)
            print(f"   -> Cleaned up dummy image: {dummy_image_path}")

    print("\n--- Email Functionality Test Finished ---")
# --- Execution Block for Local Testing ---
if __name__ == "__main__":
    print("--- Running a local test of the lambda_handler ---")
    lambda_handler(event=None, context=None)
    print("\n--- Local test finished ---")