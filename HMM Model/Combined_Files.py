import pandas as pd

# Load both CSVs
walking_df = pd.read_csv("reduced_features_labeled_Walking.csv")
standing_df = pd.read_csv("reduced_features_labeled_Standing.csv")
Running_df = pd.read_csv("reduced_features_labeled_Running.csv")
Jumping_df = pd.read_csv("reduced_features_labeled_Jumping.csv")
#Turning_Around_df = pd.read_csv("reduced_features_labeled_Turning Around.csv")

# Combine the two
combined_df = pd.concat([walking_df,standing_df,Running_df,Jumping_df], ignore_index=True)

# Save to new file
combined_df.to_csv("reduced_features_combined.csv", index=False)
print("Combined CSV saved to reduced_features_combined.csv")


