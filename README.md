# ACAMS - Automated Classroom Attendance System

### Detailed Overview of the TARP Project with ACAMS

The TARP (Time Attendance Recognition Project) aims to automate attendance systems by detecting and identifying faces from video feeds. The project leverages a combination of advanced deep learning models—YOLOv5, Faster R-CNN, and ResNet—to accurately detect and recognize faces and maintain attendance records, including names and timestamps. The system is designed to process videos by converting them into frames, detecting faces in real-time, and updating attendance databases.

## 1. Face Detection using YOLOv5
- **YOLOv5 (You Only Look Once)** is used as the primary face detection model. YOLOv5 is a real-time object detection system that can localize and identify objects (in this case, faces) in a single pass through the neural network.
- The video stream is processed frame by frame. Each frame is passed through YOLOv5, which detects faces by drawing bounding boxes around them.
- **Advantages of YOLOv5**: 
  - High speed, making it suitable for real-time face detection.
  - Good accuracy, even when the video contains multiple faces or faces in motion.

## 2. Feature Extraction using Faster R-CNN
- Once the faces are detected using YOLOv5, the bounding box coordinates are passed onto **Faster R-CNN (Region-based Convolutional Neural Network)** for further feature extraction. Faster R-CNN helps in capturing more intricate facial features, such as the shape, texture, and key identifying details.
- Faster R-CNN performs region proposal followed by classification, allowing it to extract rich feature representations of the faces from each frame.

## 3. Feature Mapping and Storage using ResNet
- The extracted features are then passed through a **ResNet (Residual Network)** model. ResNet is a powerful CNN architecture that excels at feature extraction and can handle deeper networks without losing performance due to vanishing gradients.
- The features of each detected face are mapped into a high-dimensional feature space, which helps in recognizing and distinguishing between different individuals.
- ResNet generates embeddings (or feature vectors) for each face, which are then stored and associated with the corresponding individual’s name.

## 4. Frame-by-Frame Processing
- The input video is first converted into a series of frames. Each frame is then independently passed through the YOLOv5 and Faster R-CNN models for detection and feature extraction.
- After detecting and mapping the faces, the system continuously processes the video, tracking the faces and updating attendance in real-time.

## 5. Attendance Logging with Name and Time
- Each time a face is recognized, the system checks the stored feature embeddings against previously mapped features. If a match is found, the system assigns the corresponding name to the detected face.
- **Timestamping**: The system logs the time each face is detected in the frame. This information is combined with the name to create an attendance log.
- **Attendance Record**: The final output is a comprehensive attendance record that includes the name of each individual and the corresponding time they were detected.

## 6. Workflow Summary:
1. **Video to Frames**: The video is split into individual frames.
2. **Face Detection**: YOLOv5 detects faces in each frame.
3. **Feature Extraction**: Faster R-CNN extracts detailed facial features.
4. **Feature Mapping**: ResNet stores and matches facial feature vectors.
5. **Attendance Update**: The system assigns names and logs time for each recognized face.

## 7. Applications:
- **Automated Attendance Systems**: For workplaces, schools, and events where monitoring attendance in real-time is crucial.
- **Security and Surveillance**: The system can be used to track individuals in live surveillance footage.
- **Access Control**: The system can be adapted to grant or deny access based on face recognition.

This project integrates state-of-the-art deep learning models to create a robust, real-time face detection and attendance system.
