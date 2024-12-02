import os
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import requests
import time
import serial
import threading

# Constants
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 8
MOUTH_AR_THRESH = 0.4
MOUTH_AR_CONSEC_FRAMES = 15
WARNING_YAWN_COUNT = 1
WARNING_BLINK_COUNT = 10
WARNING_DISPLAY_DURATION = 3
EYE_CLOSED_DURATION = 3

# Paths
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"
MODEL_URL = 'https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat'

# UART Configuration (Modify the UART port based on your platform)
UART_PORT = '/dev/ttyS3'  # Example: for Raspberry Pi, adjust if needed
BAUD_RATE = 9600  # Default baud rate

# Initialize counters and timers
mouth_counter = 0
total_yawns = 0
eye_counter = 0
total_blinks = 0
eye_closed_start_time = None
last_yawn_time = 0
last_eye_close_time = 0
drowsiness_detected = False
warning_reason = ""

# Track the last times a signal was sent (for limiting to 2 times every 10 seconds)
last_signal_times = []

def find_usb_cameras():
    channels = []
    for i in range(10):  # Assuming you have at most 10 video devices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Found USB camera on channel {i}")
            channels.append(i)
            cap.release()
    return channels

# Initialize serial connection
def init_serial_connection():
    try:
        ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=1)
        print(f"[INFO] Connected to {UART_PORT} at {BAUD_RATE} baud.")
        return ser
    except serial.SerialException as e:
        print(f"[ERROR] Serial connection failed: {e}")
        return None

def send_serial_data(value, num_retries=2, retry_delay=10):
    """ Send data over UART in a separate thread """
    try:
        ser = init_serial_connection()  # Open serial connection inside the function
        if ser is not None:
            byte_data = value.to_bytes(1, byteorder='little')  # Convert the value to a byte
            ser.write(byte_data)
            print(f"[INFO] Sent: {value}")
            ser.close()  # Close the connection after sending
        else:
            print("[ERROR] Serial connection is not available.")
    except Exception as e:
        print(f"[ERROR] Failed to send data: {e}")
        if num_retries > 0:
            print(f"[INFO] Retrying... {num_retries} attempts left.")
            time.sleep(retry_delay)
            send_serial_data(value, num_retries - 1, retry_delay)

# Function to download the model
def download_model(url, path):
    if not os.path.exists(path):
        print("[INFO] Downloading model...")
        response = requests.get(url)
        with open(path, 'wb') as file:
            file.write(response.content)
        print("[INFO] Model downloaded.")
    else:
        print("[INFO] Model already exists.")

# Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

# Send signal with rate-limiting to 2 times every 10 seconds
def send_signal_limited():
    global last_signal_times
    
    # Get the current time
    current_time = time.time()
    
    # Remove signal times that are older than 10 seconds
    last_signal_times = [t for t in last_signal_times if current_time - t <= 10]
    
    # Check if we have already sent 2 signals in the last 10 seconds
    if len(last_signal_times) < 2:
        threading.Thread(target=send_serial_data, args=(25,)).start()
        last_signal_times.append(current_time)  # Add current time to the list

def main():
    global mouth_counter, total_yawns, eye_counter, total_blinks, eye_closed_start_time, last_yawn_time, last_eye_close_time, drowsiness_detected, warning_reason

    print("[INFO] Loading model...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(MODEL_PATH)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    channels = find_usb_cameras()
    if not channels:
        print("No USB cameras found.")
        return

    print(f"Detected USB cameras on channels: {channels}")
    last_channel = channels[-1]
    cap = cv2.VideoCapture(last_channel)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Failed to grab frame.")
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        if len(rects) == 0:
            cv2.putText(frame, "No face detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        rect = rects[0]
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract facial landmarks for eyes and mouth
        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        mar = mouth_aspect_ratio(mouth)

        # Draw contours
        cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 0, 255), 1)

        # Handle yawning detection
        if mar > MOUTH_AR_THRESH:
            mouth_counter += 1
        else:
            if mouth_counter >= MOUTH_AR_CONSEC_FRAMES:
                total_yawns += 1
                if time.time() - last_yawn_time > 3:  # Only check yawning every 3 seconds
                    send_signal_limited()  # Use the new limited function
                    last_yawn_time = time.time()
                    drowsiness_detected = True
                    warning_reason = "Excessive Yawning"
            mouth_counter = 0

        # Handle blinking and eye closure detection
        if ear < EYE_AR_THRESH:
            eye_counter += 1
            if eye_closed_start_time is None:
                eye_closed_start_time = time.time()
        else:
            if eye_counter >= EYE_AR_CONSEC_FRAMES:
                total_blinks += 1
            eye_counter = 0
            eye_closed_start_time = None

        # Drowsiness detection logic
        current_time = time.time()

        if total_yawns >= WARNING_YAWN_COUNT:
            send_signal_limited()  # Use the new limited function
        elif eye_closed_start_time and (current_time - eye_closed_start_time >= EYE_CLOSED_DURATION):
            send_signal_limited()  # Use the new limited function
            # Display "Prolonged Eye Closure" warning message
            drowsiness_detected = True
            warning_reason = "Prolonged Eye Closure"

        # Display warning if drowsiness is detected
        if drowsiness_detected:
            cv2.putText(frame, f"Drowsiness Detected: {warning_reason}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Display yawns, blinks, EAR/MAR values on frame
        cv2.putText(frame, f"Yawns: {total_yawns}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Blinks: {total_blinks}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        #cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ensure model is available
    download_model(MODEL_URL, MODEL_PATH)
    main()

