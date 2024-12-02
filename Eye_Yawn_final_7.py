import os
import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import requests
import time
import serial
import threading
import dlib

# Constants
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 10
MOUTH_AR_THRESH = 0.5
MOUTH_AR_CONSEC_FRAMES = 3
EYE_CLOSED_DURATION = 5

# Paths
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"
MODEL_URL = 'https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat'

# Initialize counters and timers
mouth_counter = 0
total_yawns = 0
eye_counter = 0
start_time_yawn = time.time()
eye_closed_start_time = None

# UART configuration
UART_PORT = '/dev/ttyS3'  # Directly using the UART port
BAUD_RATE = 9600

def download_model(url, path):
    if not os.path.exists(path):
        print("[INFO] Downloading model...")
        response = requests.get(url)
        with open(path, 'wb') as file:
            file.write(response.content)
        print("[INFO] Model downloaded.")
    else:
        print("[INFO] Model already exists.")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4]) 
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10]) 
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

def find_usb_cameras():
    channels = []
    for i in range(10):  # Assuming you have at most 10 video devices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Found USB camera on channel {i}")
            channels.append(i)
            cap.release()
    return channels

def send_serial_data(value, times=2, interval=10):
    try:
        with serial.Serial(UART_PORT, BAUD_RATE, timeout=1) as ser:
            print(f"Connected to {UART_PORT} at {BAUD_RATE} baud.")
            ser.flushInput()
            ser.flushOutput()

            for i in range(times):
                byte_data = value.to_bytes(1, byteorder='little')  # 1 byte for 8-bit integer
                ser.write(byte_data)
                print(f"Sent: {value} ({i + 1}/{times})")
                time.sleep(interval)  # Wait for the specified interval
    except serial.SerialException as e:
        print(f"Serial Error: {e}")
    except Exception as e:
        print(f"An error occurred while sending data: {e}")

def main():
    global mouth_counter, total_yawns, eye_counter, start_time_yawn, eye_closed_start_time

    download_model(MODEL_URL, MODEL_PATH)

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

    # Track if the warning has been sent
    yawn_warning_sent = False
    eye_warning_sent = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to grab frame from camera on channel {last_channel}.")
            break

        frame = cv2.resize(frame, (640, 480))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        if len(rects) == 0:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        rect = rects[0]
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        mar = mouth_aspect_ratio(mouth)

        if mar > MOUTH_AR_THRESH:
            mouth_counter += 1
        else:
            if mouth_counter >= MOUTH_AR_CONSEC_FRAMES:
                if not yawn_warning_sent:  # Only send once per excessive yawning
                    total_yawns += 1
                    threading.Thread(target=send_serial_data, args=(15, 2, 10)).start()  # Send integer 15 over UART
                    yawn_warning_sent = True
            mouth_counter = 0
        
        if ear < EYE_AR_THRESH:
            eye_counter += 1
            if eye_closed_start_time is None:
                eye_closed_start_time = time.time()
        else:
            if eye_counter >= EYE_AR_CONSEC_FRAMES:
                eye_counter = 0
                eye_closed_start_time = None
            
        current_time = time.time()
        
        if current_time - start_time_yawn > 60:
            total_yawns = 0
            start_time_yawn = current_time
            yawn_warning_sent = False
        
        if eye_closed_start_time and (current_time - eye_closed_start_time >= EYE_CLOSED_DURATION):
            if not eye_warning_sent:  # Only send once per prolonged eye closure
                threading.Thread(target=send_serial_data, args=(20, 2, 10)).start()  # Send integer 20 over UART
                eye_warning_sent = True
            cv2.putText(frame, "WARNING: Prolonged Eye Closure!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif eye_closed_start_time is None:
            eye_warning_sent = False

        cv2.putText(frame, f"Yawns: {total_yawns}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow(f'USB Camera {last_channel}', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
