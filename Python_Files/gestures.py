import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize video capture
cap = cv2.VideoCapture(0)

# Pose classifier variables
pose = None
stage = None

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # Point A
    b = np.array(b)  # Point B
    c = np.array(c)  # Point C

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Setup Mediapipe Pose model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
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

            # Push-up Pose
            if elbow_angle < 45 and shoulder_angle > 160:
                pose_stage = "Push-up"
            # Plank Pose
            elif elbow_angle > 160 and shoulder_angle > 160 and knee_angle > 160:
                pose_stage = "Plank"
            # Sitting Cross-legged
            elif knee_angle < 90 and shoulder_angle < 90:
                pose_stage = "Sitting Cross-legged"
            # Clapping
            elif abs(shoulder[1] - wrist[1]) < 0.1 and elbow_angle < 50:
                pose_stage = "Clapping"
            # One-Hand Raised
            elif shoulder_angle < 90 and abs(wrist[1] - shoulder[1]) > 0.5:
                pose_stage = "One-Hand Raised"
            # Running
            elif abs(knee[1] - ankle[1]) > 0.2 and abs(shoulder[1] - hip[1]) > 0.2:
                pose_stage = "Running"
            # Jumping
            elif knee_angle > 160 and abs(ankle[1] - knee[1]) < 0.1:
                pose_stage = "Jumping"
            # Kicking
            elif knee_angle < 90 and ankle[1] < knee[1]:
                pose_stage = "Kicking"
            # Standing
            elif shoulder_angle > 160 and knee_angle > 160:
                pose_stage = "Standing"
            # Sitting
            elif knee_angle < 90:
                pose_stage = "Sitting"
            else:
                pose_stage = "Unknown"

            # Display the pose stage
            cv2.putText(image, pose_stage, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print("Error:", e)
            pass

        # Render pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show the image with pose classification
        cv2.imshow("Pose Classification", image)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
