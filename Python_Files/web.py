import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from Calculate_Angles import calculate_angle

# Function to reduce frame size and improve performance
def resize_frame(frame, width=640, height=480):
    return cv2.resize(frame, (width, height))

# Function for Curl Counter
def curl_counter():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    counter = 0
    stage = None
    width, height = 640, 480

    # Create a Streamlit placeholder for live video
    frame_placeholder = st.empty()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame for better performance
            frame = resize_frame(frame, width, height)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                cv2.putText(image, str(angle), tuple(np.multiply(elbow, [width, height]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 164, 143), 2, cv2.LINE_AA)

                if angle > 140:
                    stage = "down"
                if angle < 30 and stage == 'down':
                    stage = "up"
                    counter += 1

            except:
                pass

            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            cv2.putText(image, 'REPS', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,200,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Update the placeholder with the live video feed
            frame_placeholder.image(image, channels="BGR", use_column_width=True)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()

# Function for Pose Detection
def pose_detection():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    width, height = 640, 480

    # Create a Streamlit placeholder for live video
    frame_placeholder = st.empty()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame for better performance
            frame = resize_frame(frame, width, height)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Update the placeholder with the live video feed
            frame_placeholder.image(image, channels="BGR", use_column_width=True)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()

# Function for Joint Detection
def joint_detection():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    width, height = 640, 480

    # Create a Streamlit placeholder for live video
    frame_placeholder = st.empty()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame for better performance
            frame = resize_frame(frame, width, height)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                print(landmarks)
            except:
                pass

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Update the placeholder with the live video feed
            frame_placeholder.image(image, channels="BGR", use_column_width=True)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()

def gesture_recognition():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    width, height = 640, 480

    # Function to calculate angle between three points
    def calculate_angle(a, b, c):
        a = np.array(a)  # First point
        b = np.array(b)  # Middle point
        c = np.array(c)  # Last point
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    # Create a Streamlit placeholder for live video
    frame_placeholder = st.empty()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame for better performance
            frame = resize_frame(frame, width, height)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates for key points
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Calculate angles to classify pose
                shoulder_angle = calculate_angle(hip, shoulder, elbow)
                knee_angle = calculate_angle(hip, knee, ankle)
                elbow_angle = calculate_angle(shoulder, elbow, wrist)

                # Pose Detection Logic based on angle thresholds
                if elbow_angle < 45 and shoulder_angle > 160:
                    pose_stage = "Push-up"
                elif elbow_angle > 160 and shoulder_angle > 160 and knee_angle > 160:
                    pose_stage = "Plank"
                elif knee_angle < 90 and shoulder_angle < 90:
                    pose_stage = "Sitting Cross-legged"
                else:
                    pose_stage = "Unknown Pose"

                cv2.putText(image, f"Pose: {pose_stage}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            except Exception as e:
                print(e)
                pass

            # Update the placeholder with the live video feed
            frame_placeholder.image(image, channels="BGR", use_column_width=True)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    width, height = 640, 480

    def calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    # Create a Streamlit placeholder for live video
    frame_placeholder = st.empty()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame for better performance
            frame = resize_frame(frame, width, height)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates for key points
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Calculate angles to classify pose
                shoulder_angle = calculate_angle(hip, shoulder, elbow)
                knee_angle = calculate_angle(hip, knee, ankle)
                elbow_angle = calculate_angle(shoulder, elbow, wrist)

                # Pose Detection Logic
                if elbow_angle < 45 and shoulder_angle > 160:
                    pose_stage = "Push-up"
                elif elbow_angle > 160 and shoulder_angle > 160 and knee_angle > 160:
                    pose_stage = "Plank"
                elif knee_angle < 90 and shoulder_angle < 90:
                    pose_stage = "Sitting Cross-legged"
                else:
                    pose_stage = "Unknown Pose"

                cv2.putText(image, f"Pose: {pose_stage}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            except:
                pass

            # Update the placeholder with the live video feed
            frame_placeholder.image(image, channels="BGR", use_column_width=True)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()

# Streamlit interface
st.title("Live Pose Detection")

task = st.selectbox("Choose a task", ["Pose Detection", "Curl Counter", "Joint Detection", "Gesture Recognition"])

if task == "Pose Detection":
    pose_detection()
elif task == "Curl Counter":
    curl_counter()
elif task == "Joint Detection":
    joint_detection()
elif task == "Gesture Recognition":
    gesture_recognition()
