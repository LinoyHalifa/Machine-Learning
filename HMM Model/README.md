Human Motion Recognition using HMM and GMM

Overview:

This project focuses on human motion recognition using a combination of Gaussian Mixture Models (GMM) and Hidden Markov Models (HMM). The goal is to classify body movements (e.g., walking, running, standing, jumping) from pose data extracted from video frames. The model receives a new video, extracts pose keypoints using MediaPipe, converts them into motion features, and predicts the most probable sequence of actions.

How to Run the Project

1. Ensure you have Python 3.x installed, along with the required libraries:
   pip install numpy pandas scikit-learn matplotlib seaborn opencv-python mediapipe

2. Run the following scripts in order:

* extract_pose_from_video.py – extracts pose landmarks from the new video (new_video.mp4) and creates a CSV: pose_features_NewVideo.csv.

* extract_custom_features_NewVideo.py – computes geometric features from the pose CSV and saves reduced_features_NewVideo.csv.


Project Objectives

* Extract geometric features from pose landmarks

* Cluster features using GMM for discrete observations

* Train an HMM to model temporal dynamics of human motion

* Evaluate the model on unseen test data and real-world videos

Challenges

* Creating an appropriate feature representation from keypoint data

* Ensuring synchronization between pose extraction and feature computation

* Choosing the optimal number of GMM clusters for clean separation

* Building a meaningful HMM with limited data diversity

Current Accuracy

The model currently achieves ~0.82 accuracy on held-out test data. This reflects good performance on data with similar conditions to the training set.

Future Improvements

* Add more diverse video samples with different people, lighting, and angles to improve generalization

* Normalize features to reduce dependency on camera position and subject size

* Use more advanced sequence models such as LSTMs for better temporal modeling

* Automate video-to-prediction pipeline with a GUI or CLI interface

Files



1. extract_pose_features.py -Extracts pose keypoints from the movie and creats pose_features.csv files, using MediaPipe.
2. extract_custom_features.py - Reads extract_pose_features.CSV and calculating the the core distances. in addition saves the calculates and lables of the movements to reduced_features_labeled.CSV files  
3. Combined_Files.py - loads the reduced_features_labeled.CSV files from all the movements and combined them to reduced_features_combined.csv file
4. HMM_model_A5.py - The HMM model code

Feel free to open an issue or contact me if you want to contribute or improve the project!

