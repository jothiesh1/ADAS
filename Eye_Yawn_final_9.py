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
EYE_AR_CONSEC_FRAMES = 10
MOUTH_AR_THRESH = 0.4
MOUTH_AR_CONSEC_FRAMES = 15
WARNING_YAWN_COUNT = 1
WARNING_BLINK_COUNT = 10
WARNING_DISPLAY_DURATION = 3
EYE_CLOSED_DURATION = 4
COUNTER_RESET_INTERVAL = 5

# Paths
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"
MODEL_URL = 'https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat'

# Initialize counters and timers
mouth_counter = 0
total_yawns = 0
eye_counter = 0
total_blinks = 0
start_time_yawn = time.time()
start_time_blink = time.time()
warning_time = None
eye_closed_start_time = None
drowsiness_detected = False
warning_reason = ""

# Flags to track if warnings have been sent
yawn_warning_sent = False
eye_closure_warning_sent = False

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

def main():
    global mouth_counter, total_yawns, eye_counter, total_blinks, start_time_yawn, start_time_blink, warning_time, eye_closed_start_time, drowsiness_detected, warning_reason, yawn_warning_sent, eye_closure_warning_sent
    
    print("[INFO] Loading model...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(MODEL_PATH)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    cap = cv2.VideoCapture(0)

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
        
        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        mar = mouth_aspect_ratio(mouth)

        cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 0, 255), 1)

        if mar > MOUTH_AR_THRESH:
            mouth_counter += 1
        else:
            if mouth_counter >= MOUTH_AR_CONSEC_FRAMES:
                total_yawns += 1
            mouth_counter = 0

        if ear < EYE_AR_THRESH:
            eye_counter += 1
            if eye_closed_start_time is None:
                eye_closed_start_time = time.time()
        else:
            if eye_counter >= EYE_AR_CONSEC_FRAMES:
                total_blinks += 1
            eye_counter = 0
            eye_closed_start_time = None

        current_time = time.time()
        if current_time - start_time_yawn >= COUNTER_RESET_INTERVAL:
            total_yawns = 0
            start_time_yawn = current_time
            yawn_warning_sent = False

        if current_time - start_time_blink >= COUNTER_RESET_INTERVAL:
            total_blinks = 0
            start_time_blink = current_time

        if total_yawns >= WARNING_YAWN_COUNT and not yawn_warning_sent:
            drowsiness_detected = True
            warning_reason = "Excessive Yawning"
            threading.Thread(target=send_serial_data, args=(25, 2, 10)).start()  # Send integer 25 over UART
            warning_time = current_time
            yawn_warning_sent = True
        elif eye_closed_start_time and (current_time - eye_closed_start_time >= EYE_CLOSED_DURATION) and not eye_closure_warning_sent:
            drowsiness_detected = True
            warning_reason = "Prolonged Eye Closure"
            threading.Thread(target=send_serial_data, args=(20, 2, 10)).start()  # Send integer 20 over UART
            warning_time = current_time
            eye_closure_warning_sent = True
        else:
            drowsiness_detected = False
            warning_reason = ""

        if warning_time and (current_time - warning_time <= WARNING_DISPLAY_DURATION):
            cv2.putText(frame, f"Drowsiness Detected: {warning_reason}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        elif warning_time and (current_time - warning_time > WARNING_DISPLAY_DURATION):
            warning_time = None
            warning_reason = ""

        cv2.putText(frame, f"Yawns: {total_yawns}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Blinks: {total_blinks}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
