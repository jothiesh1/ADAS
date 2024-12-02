import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import serial
import time
import requests
import os

# Constants for facial features detection
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 15
MOUTH_AR_THRESH = 0.4  # Lowered threshold for yawning detection
MOUTH_AR_CONSEC_FRAMES = 20  
WARNING_BLINK_COUNT = 10
EYE_CLOSED_DURATION = 5  # seconds

# UART Configuration  
UART_PORT = '/dev/ttyS3'  # Modify based on your system
BAUD_RATE = 9600

# Signal values
YAWN_SIGNAL = 20
EYE_CLOSURE_SIGNAL = 25

class DrowsinessDetector:
    def __init__(self):
        # Initialize facial landmark detector
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = self.load_predictor()
        
        # Get facial landmarks indices
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        
        # Initialize counters and timers
        self.blink_counter = 0
        self.yawn_counter = 0
        self.total_blinks = 0 
        self.total_yawns = 0
        self.eye_closed_start = None
        self.last_yawn_time = 0
        self.last_eye_closure_time = 0
        self.eye_closure_signaled = False  # Flag to track if eye closure signal is already sent
        
        # Initialize UART
        self.setup_uart()

    def load_predictor(self):
        model_path = 'shape_predictor_68_face_landmarks.dat'
        if not os.path.exists(model_path):
            print("[INFO] Downloading facial landmark predictor...")
            url = 'https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat'
            response = requests.get(url)
            with open(model_path, 'wb') as f:
                f.write(response.content)
        return dlib.shape_predictor(model_path)

    def setup_uart(self):
        try:
            self.serial_conn = serial.Serial(UART_PORT, BAUD_RATE, timeout=1)
            print(f"[INFO] UART initialized on {UART_PORT}")
        except Exception as e:
            print(f"[ERROR] UART initialization failed: {e}")
            self.serial_conn = None

    def send_uart_signal(self, signal_value):
        if self.serial_conn:
            try:
                self.serial_conn.write(signal_value.to_bytes(1, 'little'))
                print(f"[INFO] Sent signal: {signal_value}")
            except Exception as e:
                print(f"[ERROR] Failed to send UART signal: {e}")
        else:
            print("[ERROR] UART connection is not initialized.")

    def calculate_ear(self, eye):
        # Calculate eye aspect ratio
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def calculate_mar(self, mouth):
        # Calculate mouth aspect ratio  
        A = dist.euclidean(mouth[2], mouth[10]) 
        B = dist.euclidean(mouth[4], mouth[8])
        C = dist.euclidean(mouth[0], mouth[6])
        return (A + B) / (2.0 * C)

    def process_frame(self, frame):
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)

        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.eye_closure_signaled = False  # Reset eye closure signal flag
            return frame

        face = faces[0]    
        shape = face_utils.shape_to_np(self.predictor(gray, face))

        # Extract eye and mouth features
        left_eye = shape[self.lStart:self.lEnd]
        right_eye = shape[self.rStart:self.rEnd]
        mouth = shape[self.mStart:self.mEnd]

        # Calculate ratios
        ear = (self.calculate_ear(left_eye) + self.calculate_ear(right_eye)) / 2.0
        mar = self.calculate_mar(mouth)

        # Draw features
        cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)  
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 0, 255), 1)

        # Process eye closure
        current_time = time.time()
        if ear < EYE_AR_THRESH:
            self.blink_counter += 1
            if self.eye_closed_start is None:
                self.eye_closed_start = current_time
            
            # Check for prolonged eye closure
            if self.eye_closed_start and (current_time - self.eye_closed_start >= EYE_CLOSED_DURATION) and not self.eye_closure_signaled:
                self.send_uart_signal(EYE_CLOSURE_SIGNAL)  # Send prolonged eye closure signal
                self.last_eye_closure_time = current_time
                self.eye_closure_signaled = True  # Set flag to avoid repeated signals
                cv2.putText(frame, "WARNING: Prolonged Eye Closure!", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            if self.blink_counter >= EYE_AR_CONSEC_FRAMES:
                self.total_blinks += 1
            self.blink_counter = 0
            self.eye_closed_start = None
            self.eye_closure_signaled = False  # Reset eye closure signal flag

        # Process yawning  
        if mar > MOUTH_AR_THRESH:
            self.yawn_counter += 1
            if self.yawn_counter >= MOUTH_AR_CONSEC_FRAMES:
                if current_time - self.last_yawn_time >= 3.0:  # Minimum 3 seconds between yawn signals
                    print("[INFO] Yawn detected!")
                    self.send_uart_signal(YAWN_SIGNAL)  # Send yawning signal
                    self.total_yawns += 1
                    self.last_yawn_time = current_time
                cv2.putText(frame, "YAWN DETECTED!", (10, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            self.yawn_counter = 0

        # Display counters  
        cv2.putText(frame, f"Blinks: {self.total_blinks}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Yawns: {self.total_yawns}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame

    def find_usb_cameras(self):
        channels = []
        for i in range(10):  # Assuming you have at most 10 video devices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Found USB camera on channel {i}")
                channels.append(i)
                cap.release()
        return channels

if __name__ == "__main__":
    detector = DrowsinessDetector()

    # Check and find USB cameras 
    channels = detector.find_usb_cameras()
    if not channels:
        print("No USB cameras found.")
    else:
        print(f"Detected USB cameras on channels: {channels}")
        last_channel = channels[-1]  # Using the last detected USB camera
        cap = cv2.VideoCapture(last_channel)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame from USB camera")
                break

            frame = detector.process_frame(frame)
            cv2.imshow("Drowsiness Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows() 
        if detector.serial_conn:
            detector.serial_conn.close()


