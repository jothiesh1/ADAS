import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
import dlib
import time
import serial
import threading
import warnings
from scipy.spatial import distance as dist
from imutils import face_utils
import requests
import os

warnings.filterwarnings("ignore")

# Constants
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 8
MOUTH_AR_THRESH = 0.4
MOUTH_AR_CONSEC_FRAMES = 20
WARNING_YAWN_COUNT = 1
WARNING_BLINK_COUNT = 10
WARNING_DISPLAY_DURATION = 3
EYE_CLOSED_DURATION = 5

# Paths and URLs
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"
MODEL_URL = 'https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat'
UART_PORT = '/dev/ttyS3'  # Update based on your platform
BAUD_RATE = 9600

# Initialize counters and timers
mouth_counter = 0
total_yawns = 0
eye_counter = 0
total_blinks = 0
eye_closed_start_time = None
warning_time = None
drowsiness_detected = False
warning_reason = ""

# Flag for prolonged eye closure detection
prolonged_eye_closure_triggered = False

# Reset timers
last_reset_time = time.time()

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
        # Optionally handle retries
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

# Extract facial features using MediaPipe
def extract_features(img, face_mesh):
    NOSE = 1
    FOREHEAD = 10
    LEFT_EYE = 33
    MOUTH_LEFT = 61
    CHIN = 199
    RIGHT_EYE = 263
    MOUTH_RIGHT = 291

    result = face_mesh.process(img)
    face_features = []
    
    if result.multi_face_landmarks != None:
        for face_landmarks in result.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [FOREHEAD, NOSE, MOUTH_LEFT, MOUTH_RIGHT, CHIN, LEFT_EYE, RIGHT_EYE]:
                    face_features.append(lm.x)
                    face_features.append(lm.y)

    return face_features

# Normalize facial features
def normalize(poses_df):
    normalized_df = poses_df.copy()
    
    for dim in ['x', 'y']:
        # Centering around the nose
        for feature in ['forehead_'+dim, 'nose_'+dim, 'mouth_left_'+dim, 'mouth_right_'+dim, 'left_eye_'+dim, 'chin_'+dim, 'right_eye_'+dim]:
            normalized_df[feature] = poses_df[feature] - poses_df['nose_'+dim]
        
        # Scaling
        diff = normalized_df['mouth_right_'+dim] - normalized_df['left_eye_'+dim]
        for feature in ['forehead_'+dim, 'nose_'+dim, 'mouth_left_'+dim, 'mouth_right_'+dim, 'left_eye_'+dim, 'chin_'+dim, 'right_eye_'+dim]:
            normalized_df[feature] = normalized_df[feature] / diff
    
    return normalized_df

# Draw axes on the face (pitch, yaw, roll) 
def draw_axes(img, pitch, yaw, roll, tx, ty, size=50):
    yaw = -yaw
    rotation_matrix = cv2.Rodrigues(np.array([pitch, yaw, roll]))[0].astype(np.float64)
    axes_points = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=np.float64)
    axes_points = rotation_matrix @ axes_points
    axes_points = (axes_points[:2, :] * size).astype(int)
    axes_points[0, :] = axes_points[0, :] + tx
    axes_points[1, :] = axes_points[1, :] + ty
    
    new_img = img.copy()
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 0].ravel()), (255, 0, 0), 3)    
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 1].ravel()), (0, 255, 0), 3)    
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 2].ravel()), (0, 0, 255), 3)
    return new_img

def find_usb_cameras():
    channels = []
    for i in range(10):  # Assuming you have at most 10 video devices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Found USB camera on channel {i}")
            channels.append(i)
            cap.release()
    return channels

# Main Program
print("[INFO] Loading model...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PATH)

# Load prediction model (replace with actual model path)
model = pickle.load(open('./model.pkl', 'rb'))

# Download the shape predictor model if it doesn't exist
download_model(MODEL_URL, MODEL_PATH)

# MediaPipe FaceMesh initialization
face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)


channels = find_usb_cameras()
# if not channels:
#     print("No USB cameras found.")
#     return

print(f"Detected USB cameras on channels: {channels}")
last_channel = channels[-1]
print(f'---------------{last_channel}-----------')
cap = cv2.VideoCapture(last_channel)
# Initialize video capture
#cap = cv2.VideoCapture(0)

frame_skip = 1  # Process every 3rd frame
frame_count = 0


while(cap.isOpened()):
    ret, img = cap.read()
    if ret:

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 1)
        img_h, img_w, img_c = img.shape
        text = ''

        face_features = extract_features(img, face_mesh)
        if len(face_features):
            face_features_df = pd.DataFrame([face_features], columns=[f"{pos}_{dim}" for pos in ['nose', 'forehead', 'left_eye', 'mouth_left', 'chin', 'right_eye', 'mouth_right'] for dim in ('x', 'y')])
            face_features_normalized = normalize(face_features_df)
            pitch_pred, yaw_pred, roll_pred = model.predict(face_features_normalized).ravel()

            # Calculate nose position to draw axes
            nose_x = face_features_df['nose_x'].values * img_w
            nose_y = face_features_df['nose_y'].values * img_h
            img = draw_axes(img, pitch_pred, yaw_pred, roll_pred, nose_x, nose_y)

            # Interpret the head pose (pitch, yaw, roll)
            if pitch_pred > 0.4:
                text = 'Top'
                if yaw_pred > 0.4:
                    text = 'Top Left'
                elif yaw_pred < -0.4:
                    text = 'Top Right'
            elif pitch_pred < -0.4:
                text = 'Bottom'
                if yaw_pred > 0.4:
                    text = 'Bottom Left'
                elif yaw_pred < -0.4:
                    text = 'Bottom Right'
            elif yaw_pred > 0.6:
                text = 'Left'
            elif yaw_pred < -0.6:
                text = 'Right'
            else:
                text = 'Forward'

                # Process face for yawning and blinking
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                rects = detector(gray, 0)
                if len(rects) > 0:
                    rect = rects[0]
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
    
                    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
                    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
                    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    
                    left_eye = shape[lStart:lEnd]
                    right_eye = shape[rStart:rEnd]
                    mouth = shape[mStart:mEnd]
    
                    left_ear = eye_aspect_ratio(left_eye)
                    right_ear = eye_aspect_ratio(right_eye)
                    ear = (left_ear + right_ear) / 2.0
                    mar = mouth_aspect_ratio(mouth)

                    # Yawning detection
                    if mar > MOUTH_AR_THRESH:
                        mouth_counter += 1
                    else:
                        if mouth_counter >= MOUTH_AR_CONSEC_FRAMES:
                            total_yawns += 1
                            if not drowsiness_detected:
                                threading.Thread(target=send_serial_data, args=(15,)).start()  # Send 15 when yawning is detected
                                drowsiness_detected = True
                                warning_reason = "Excessive Yawning"
                        mouth_counter = 0
    
                    # Blinking detection
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

                    # Reset counters every 15 seconds
                    if current_time - last_reset_time > 15:
                        mouth_counter = 0
                        eye_counter = 0
                        total_yawns = 0
                        total_blinks = 0
                        drowsiness_detected = False
                        warning_reason = ""
                        prolonged_eye_closure_triggered = False
                        last_reset_time = current_time

                    if total_yawns >= WARNING_YAWN_COUNT:
                        drowsiness_detected = True
                        warning_reason = "Excessive Yawning"
                        warning_time = current_time
                    elif total_blinks >= WARNING_BLINK_COUNT:
                        drowsiness_detected = True
                        warning_reason = "Excessive Blinking"
                        warning_time = current_time
                    elif eye_closed_start_time and (current_time - eye_closed_start_time >= EYE_CLOSED_DURATION):
                        if not prolonged_eye_closure_triggered:
                            prolonged_eye_closure_triggered = True
                            drowsiness_detected = True
                            warning_reason = "Prolonged Eye Closure"
                            warning_time = current_time
                            # Send serial data when prolonged eye closure is detected
                            threading.Thread(target=send_serial_data, args=(25,)).start()  # Send 25 when prolonged eye closure is detected
                    else:
                        drowsiness_detected = False
                        warning_reason = ""
    
                    # Display warning if needed
                    if warning_time and (current_time - warning_time <= WARNING_DISPLAY_DURATION):
                        cv2.putText(img, f"Drowsiness Detected: {warning_reason}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    elif warning_time and (current_time - warning_time > WARNING_DISPLAY_DURATION):
                        warning_time = None
                        warning_reason = ""
    
                    # Display yawns, blinks, EAR/MAR values on frame
                    cv2.putText(img, f"Yawns: {total_yawns}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(img, f"MAR: {mar:.2f}", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(img, f"Blinks: {total_blinks}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(img, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display head pose direction
            cv2.putText(img, text, (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('img', img)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
