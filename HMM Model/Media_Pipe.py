import cv2
import mediapipe as mp

# Initialize MediaPipe Pose and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Path to your video file
video_path = 'Man Walking in Nature.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties for saving output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
output_path = 'output_pose_estimation.avi'

# Define VideoWriter to save the output video
out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*'XVID'),
    fps,
    (frame_width, frame_height)
)

# Initialize the pose model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and get landmarks
        results = pose.process(frame_rgb)

        # Draw the landmarks and connections on the frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
            )

        # Show the annotated frame
        cv2.imshow('Pose Estimation', frame)

        # Save the annotated frame to the output video
        out.write(frame)

        # Press ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Release everything after processing
cap.release()
out.release()
cv2.destroyAllWindows()
