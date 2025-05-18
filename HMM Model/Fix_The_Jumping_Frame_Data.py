import pandas as pd


# Load each CSV
jumping_df = pd.read_csv("reduced_features_labeled_Jumping.csv")

# Drop extra columns to match others
columns_to_keep = [
    'ankle_distance', 'shoulder_distance', 'hip_angle',
    'ankle_movement', 'ankle_speed', 'ankle_to_hip_ratio',
    'hip_distance', 'knee_distance', 'shoulder_to_ankle_ratio', 'label'
]
jumping_df = jumping_df[columns_to_keep]

# Save the cleaned file
jumping_df.to_csv("reduced_features_labeled_Jumping.csv", index=False)
