# Human Pose Estimation Using Machine Learning

This project demonstrates human pose estimation using MediaPipe and OpenCV. The program processes static images and real-time webcam input to detect human poses and landmarks.

## Features
- Pose estimation for static images and real-time webcam input.
- Annotated pose landmarks and segmentation masks.
- Nose coordinates logging for detected landmarks.
- Output annotated images saved in a dedicated folder.

## Technologies Used
- Python
- MediaPipe
- OpenCV
- NumPy

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Allminfatima/Human-Pose-Estimation.git
   cd Human-Pose-Estimation
   
2. Install the required dependencies:
   ```bash
   pip install opencv-python mediapipe numpy

3. Place your static images/videos in the input_files/ folder (create if necessary).

# Usage

## For Static Images:
1. Replace the file paths in the IMAGE_FILES variable with your image or video file paths.
2. Run the script
3. Annotated images will be saved in the output_images/ folder.

## For Webcam Input:
1. Ensure your webcam is connected.
2. Run the script:
3. The live feed with pose annotations will appear. Press Esc to exit.

# Output


# Future Enhancements
1. Support for video input files.
2. Exporting pose landmarks as JSON or CSV for analysis.
3. Integration with deep learning models for advanced pose estimation.
