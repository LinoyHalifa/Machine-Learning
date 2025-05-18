import cv2
import mediapipe as mp
import pandas as pd

# MediaPipe pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

# Load the video
video_path = "Man Walking in Nature.mp4"
cap = cv2.VideoCapture(video_path)

data = []
frame_num = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        row = [frame_num]
        for lm in landmarks:
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
        data.append(row)

    frame_num += 1

cap.release()
pose.close()

# Build Columns
columns = ["frame"]
for i in range(33):  # There are 33 points in the body
    columns += [f"x_{i}", f"y_{i}", f"z_{i}", f"v_{i}"]

df = pd.DataFrame(data, columns=columns)
df.to_csv("pose_features.csv", index=False)

print(" Data saved to pose_features.csv")
