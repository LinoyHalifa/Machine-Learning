import pandas as pd
import numpy as np

# Load the full pose landmarks data
df = pd.read_csv("pose_features_Turning Around.csv")

# Function to compute Euclidean distance
def euclidean(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Function to compute hip angle
def compute_hip_angle(x1, y1, x2, y2):
    vec = np.array([x2 - x1, y2 - y1])
    vertical = np.array([0, -1])
    unit_vec = vec / np.linalg.norm(vec)
    dot_product = np.dot(unit_vec, vertical)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle)

features = []
previous_ankle_distance = None
frame_time = 1 / 30  # 30 FPS

for i, row in df.iterrows():
    feature_row = {}

    # Core distances
    ankle_dist = euclidean(row["x_27"], row["y_27"], row["x_28"], row["y_28"])
    shoulder_dist = euclidean(row["x_11"], row["y_11"], row["x_12"], row["y_12"])
    hip_to_knee = euclidean(row["x_24"], row["y_24"], row["x_26"], row["y_26"])

    # Feature 1: ankle_distance
    feature_row["ankle_distance"] = ankle_dist

    # Feature 2: shoulder_distance
    feature_row["shoulder_distance"] = shoulder_dist

    # Feature 3: hip_angle (right leg)
    feature_row["hip_angle"] = compute_hip_angle(row["x_24"], row["y_24"], row["x_26"], row["y_26"])

    # Feature 4: ankle_movement
    if previous_ankle_distance is None:
        feature_row["ankle_movement"] = 0
    else:
        feature_row["ankle_movement"] = abs(ankle_dist - previous_ankle_distance)
    previous_ankle_distance = ankle_dist

    # Feature 5: ankle_speed
    feature_row["ankle_speed"] = feature_row["ankle_movement"] / frame_time

    # Feature 6: ankle_to_hip_ratio
    feature_row["ankle_to_hip_ratio"] = ankle_dist / hip_to_knee if hip_to_knee != 0 else 0

    # Feature 7: hip_distance (between right and left hips: 23 and 24)
    feature_row["hip_distance"] = euclidean(row["x_23"], row["y_23"], row["x_24"], row["y_24"])

    # Feature 8: knee_distance (between knees: 25 and 26)
    feature_row["knee_distance"] = euclidean(row["x_25"], row["y_25"], row["x_26"], row["y_26"])

    # Feature 9: shoulder_to_ankle_ratio
    feature_row["shoulder_to_ankle_ratio"] = shoulder_dist / ankle_dist if ankle_dist != 0 else 0

    features.append(feature_row)

# Save new features
df_features = pd.DataFrame(features)
df_features.to_csv("reduced_features_Turning Around.csv", index=False)
print("reduced_features_Turning Around.csv saved!")

# Add label
df_features["label"] = "Turning Around"
df_features.to_csv("reduced_features_labeled_Turning Around.csv", index=False)
print("Labels added to reduced_features_labeled_Turning Around.csv")
