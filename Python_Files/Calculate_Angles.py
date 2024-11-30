import numpy as np

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # Point A
    b = np.array(b)  # Point B
    c = np.array(c)  # Point C

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 160.0:
        angle = 360 - angle
    return angle

# Example usage with landmarks
shoulder = [0.5, 0.5]  # Replace with actual landmark data
elbow = [0.6, 0.6]     # Replace with actual landmark data
wrist = [0.7, 0.7]     # Replace with actual landmark data

# Calculate angle
angle = calculate_angle(shoulder, elbow, wrist)
print(f"Angle: {angle}")
