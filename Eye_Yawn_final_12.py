import cv2
import numpy as np
import mediapipe as mp
import time
import serial
import threading

# Load the YOLOv3-Tiny model for faster detection
net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')

# Read class names
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
phone_class_id = classes.index('cell phone')

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

def find_usb_cameras():
    channels = []
    for i in range(10):  # Assuming you have at most 10 video devices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Found USB camera on channel {i}")
            channels.append(i)
            cap.release()
    return channels

channels = find_usb_cameras()
if not channels:
    print("No USB cameras found.")
    

print(f"Detected USB cameras on channels: {channels}")
last_channel = channels[-1]
cap = cv2.VideoCapture(last_channel)

# Initialize MediaPipe Face Mesh for drowsiness and yawning detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Enables iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Constants for drowsiness detection
EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold
EAR_CONSEC_FRAMES = 15  # Number of consecutive frames to check
MAR_THRESHOLD = 0.75  # Mouth Aspect Ratio threshold
MAR_CONSEC_FRAMES = 20  # Number of consecutive frames to check
DROWSINESS_DURATION = 3  # Duration of continuous drowsiness to trigger warning (in seconds)

# Constants for face orientation detection
ORIENTATION_RATIO_THRESHOLD = 0.7  # Threshold for significant face turns
ORIENTATION_CONSEC_FRAMES = 15  # Number of consecutive frames to check

# **Updated: Phone detection confidence threshold**
PHONE_CONFIDENCE_THRESHOLD = 0.2  # Decreased from 0.4 to 0.2

# Initialize counters
drowsy_counter = 0
yawn_counter = 0
orientation_counter = 0

# Initialize flags for sending signals
drowsy_signal_sent = False
yawn_signal_sent = False

# Initialize drowsiness timer
drowsy_start_time = None

# Indices for eyes landmarks in MediaPipe Face Mesh
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Indices for eye centers and nose
LEFT_EYE_CENTER_INDICES = [33, 133]  # Outer corners of left eye
RIGHT_EYE_CENTER_INDICES = [362, 263]  # Outer corners of right eye
NOSE_TIP_INDEX = 1  # Nose tip

def send_serial_data(value, times=1, interval=10):
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


def compute_ear(landmarks, eye_indices):
    # Assign landmarks
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]
    # Compute EAR
    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    C = np.linalg.norm(p1 - p4)
    ear = (A + B) / (2.0 * C)
    return ear

def compute_mar(landmarks):
    # Assign landmarks for MAR calculation
    p13 = landmarks[13]    # Upper lip
    p14 = landmarks[14]    # Lower lip
    p78 = landmarks[78]    # Left inner lip corner
    p308 = landmarks[308]  # Right inner lip corner
    p61 = landmarks[61]    # Left outer lip corner
    p291 = landmarks[291]  # Right outer lip corner
    # Compute vertical distances
    A = np.linalg.norm(p13 - p14)
    B = np.linalg.norm(p78 - p308)
    # Compute horizontal distance
    C = np.linalg.norm(p61 - p291)
    # Compute MAR
    mar = (A + B) / (2.0 * C)
    return mar

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()
    height, width, _ = frame.shape

    # Prepare blob and perform forward pass for phone detection
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialization
    class_ids = []
    confidences = []
    boxes = []

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # **Updated: Use the new confidence threshold**
            if confidence > PHONE_CONFIDENCE_THRESHOLD and class_id == phone_class_id:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # Non-max suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, PHONE_CONFIDENCE_THRESHOLD, 0.3)

    # Draw bounding boxes if any detections are made
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = f"Cell Phone: {confidences[i]:.2f}"
            color = (0, 255, 0)  # Green color
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            print(f"Detected 'cell phone' with confidence {confidences[i]:.2f}")  # Debugging

    # Convert the BGR image to RGB before processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        # Collect landmark points
        landmarks = []
        for lm in face_landmarks.landmark:
            x, y = int(lm.x * width), int(lm.y * height)
            landmarks.append([x, y])

        landmarks = np.array(landmarks, dtype=np.float64)

        # Calculate EAR for both eyes
        left_ear = compute_ear(landmarks, LEFT_EYE_INDICES)
        right_ear = compute_ear(landmarks, RIGHT_EYE_INDICES)
        ear = (left_ear + right_ear) / 2.0

        # Calculate MAR for mouth
        mar = compute_mar(landmarks)

        # Display EAR and MAR values (optional, for debugging)
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Check for drowsiness
        if ear < EAR_THRESHOLD:
            if drowsy_start_time is None:
                drowsy_start_time = time.time()  # Start the timer
            
            drowsy_duration = time.time() - drowsy_start_time
            if drowsy_duration >= DROWSINESS_DURATION and not drowsy_signal_sent:
                threading.Thread(target=send_serial_data, args=(25,)).start()  # Send integer 25 over UART once
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                drowsy_signal_sent = True
        else:
            drowsy_start_time = None
            drowsy_signal_sent = False

        # Check for yawning
        if mar > MAR_THRESHOLD:
            yawn_counter += 1
            if yawn_counter >= MAR_CONSEC_FRAMES and not yawn_signal_sent:
                threading.Thread(target=send_serial_data, args=(20,)).start()  # Send integer 20 over UART once
                cv2.putText(frame, "YAWNING DETECTED", (10, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                yawn_signal_sent = True
        else:
            yawn_counter = 0
            yawn_signal_sent = False

        # Face orientation detection using eye and nose positions
        # Get left and right eye centers
        left_eye_center = np.mean(landmarks[LEFT_EYE_CENTER_INDICES], axis=0)
        right_eye_center = np.mean(landmarks[RIGHT_EYE_CENTER_INDICES], axis=0)
        # Get nose tip position
        nose_tip = landmarks[NOSE_TIP_INDEX]

        # Calculate horizontal distances
        dist_left = abs(nose_tip[0] - left_eye_center[0])
        dist_right = abs(nose_tip[0] - right_eye_center[0])

        # Calculate ratio
        ratio = dist_left / dist_right if dist_right != 0 else 0

        # Display ratio (optional, for debugging)
        cv2.putText(frame, f"Orientation Ratio: {ratio:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Check for face orientation
        if abs(1 - ratio) > ORIENTATION_RATIO_THRESHOLD:
            orientation_counter += 1
            if orientation_counter >= ORIENTATION_CONSEC_FRAMES:
                cv2.putText(frame, "FACE ORIENTATION ALERT!", (10, 270),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            orientation_counter = 0

        # Draw facial landmarks (optional)
        mp_drawing.draw_landmarks(
            frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1))

    else:
        # Reset counters if no face is detected
        drowsy_start_time = None
        drowsy_signal_sent = False
        yawn_counter = 0
        yawn_signal_sent = False
        orientation_counter = 0

    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Display the frame
    #cv2.imshow('Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
