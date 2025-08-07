# Bald Eagle Detection Bot 🦅

This project is a Python-based application that uses computer vision to monitor the [Big Bear Bald Eagles Nest YouTube livestream](https://www.youtube.com/watch?v=B4-L2nfGcuE). When a bald eagle is detected, the bot automatically saves an annotated screenshot and sends an email alert with the image attached.


*(Placeholder for image)*

## The Story Behind the Project

This project started as an exploration into real-world application of machine learning and a desire to build a cost-effective, serverless monitoring tool. My initial goal was to deploy the bot on AWS Lambda. This involved a significant effort to package heavy libraries like TensorFlow and OpenCV into a Lambda-compatible Docker container.

After successfully deploying to the cloud, I encountered a roadblock that code couldn't easily solve: YouTube's anti-bot measures. Running from an AWS datacenter IP with no browser cookies, the script was flagged and blocked. After a brief hiatus from the project, I pivoted to a more pragmatic and ultimately more reliable solution: hosting the script on a dedicated spare computer. This `README` documents the final, working version of that approach.

## Features

- **Live Stream Monitoring:** Uses `streamlink` to reliably connect to and process a YouTube livestream.
- **AI-Powered Detection:** Employs a pre-trained ResNet50 model with TensorFlow/Keras to identify bald eagles in the video feed.
- **Automated Alerts:** Sends an email notification via Gmail's SMTP server upon a positive detection.
- **Visual Proof:** Captures a frame from the livestream, annotates it with the detection probability, and attaches it to the notification email.
- **Continuous Operation:** Designed to run 24/7 as a background service on a dedicated machine.

## Tech Stack

- **Language:** Python 3.13
- **Computer Vision:** OpenCV
- **Machine Learning:** TensorFlow / Keras
- **Stream Handling:** Streamlink
- **Email & Environment:** smtplib, python-dotenv

---

## Setup and Installation

This guide assumes you are setting this up to run continuously on a dedicated Windows machine.

### Step 1: Clone the Repository

First, clone this repository to a permanent location on your machine (e.g., `C:\Projects\EagleBot`).

```bash
git clone https://github.com/ethandillon/Bald-Eagle-Recognition
cd your-repo-name
```

### Step 2: Install Python

If you don't have Python installed, download it from [python.org](https://www.python.org/downloads/). During installation, **make sure that you check the box that says "Add Python to PATH"**.

### Step 3: Set Up the Environment

1.  **Create a Virtual Environment:** Open a Command Prompt in the project directory and create a virtual environment to keep dependencies isolated.
    ```cmd
    python -m venv .venv
    ```

2.  **Activate the Virtual Environment:**
    ```cmd
    .venv\Scripts\activate
    ```
    Your command prompt should now be prefixed with `(.venv)`.

3.  **Install Required Packages:** Use the provided `requirements.txt` file to install all necessary libraries.
    ```cmd
    pip install -r requirements.txt
    ```

### Step 4: Configuration

You need to create a `.env` file to securely store your email password.

1.  Create a file named `.env` in the root of the project directory.
2.  Generate a 16-digit **Google App Password** for your sender Gmail account (your regular password will not work). You must have 2-Factor Authentication enabled to do this. [Follow these instructions](https://support.google.com/accounts/answer/185833).
3.  Add the App Password to your `.env` file like this:
    ```
    EMAIL_PASSWORD='your16digitapppasswordgoeshere'
    ```

### Step 5: Test the Script

Before automating the script, run it manually from your activated virtual environment to ensure everything is working.

```cmd
python app.py
```

The script should start, load the model, and begin its first detection cycle. You can use `Ctrl+C` to stop it.

---

## Automation on Windows

To make this script run 24/7 and automatically start on boot, we will use the Windows Task Scheduler.

1.  **Open Task Scheduler** and click `Create Task...`.
2.  **General Tab:**
    - Name: `Bald Eagle Detection Bot`
    - Select `Run whether user is logged on or not`.
3.  **Triggers Tab:**
    - Click `New...` and set `Begin the task:` to `At startup`.
    - **Important:** Do NOT set it to repeat. The Python script handles its own timing loop.
4.  **Actions Tab (Critical):**
    - Click `New...` and set Action to `Start a program`.
    - **Program/script:** Provide the **full, absolute path** to the Python executable in your virtual environment. Example:
      `C:\Projects\EagleBot\.venv\Scripts\python.exe`
    - **Add arguments:** `app.py`
    - **Start in:** Provide the **full, absolute path** to your project folder. Example:
      `C:\Projects\EagleBot`
5.  **Settings Tab:**
    - **Uncheck** `Stop the task if it runs longer than:`.
6.  **Save the Task.** It will prompt for your Windows password to grant permissions.

### Important Note on Laptop Setup
If running on a laptop, you must configure its power settings to **never sleep** when plugged in and to **do nothing** when the lid is closed. This ensures the script runs uninterrupted. Be mindful of heat and ensure proper ventilation.

---
```
