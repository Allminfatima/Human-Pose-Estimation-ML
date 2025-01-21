# Import necessary packages
import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize Mediapipe Pose and Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# File path for static image/video
IMAGE_FILES = [r"I:\programmer 1\programmer 2.1\pexels-roman-odintsov-8233009.mp4"]
OUTPUT_DIR = r"output_images"  # Directory to save annotated images
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure the output directory exists

BG_COLOR = (192, 192, 192)  # Background color for segmentation

# For processing static images
with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:
    for idx, file_path in enumerate(IMAGE_FILES):
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            continue
        
        image = cv2.imread(file_path)
        image_height, image_width, _ = image.shape
        
        # Process the image
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.pose_landmarks:
            print(f"No landmarks detected in {file_path}")
            continue

        # Log nose coordinates
        nose_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width
        nose_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height
        print(f'Nose coordinates: ({nose_x}, {nose_y})')

        # Annotate the image
        annotated_image = image.copy()
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        annotated_image = np.where(condition, annotated_image, bg_image)
        
        # Draw landmarks
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        output_path = os.path.join(OUTPUT_DIR, f'annotated_image_{idx}.png')
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved at: {output_path}")

        # Plot pose landmarks in 3D
        mp_drawing.plot_landmarks(
            results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

# For webcam input
cap = cv2.VideoCapture(0)  # Use 0 for default camera
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Process the frame
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame)

        # Annotate the frame
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        # Display the frame
        cv2.imshow('MediaPipe Pose', cv2.flip(frame, 1))
        if cv2.waitKey(5) & 0xFF == 27:  # Exit on 'Esc' key
            break
cap.release()
cv2.destroyAllWindows()
