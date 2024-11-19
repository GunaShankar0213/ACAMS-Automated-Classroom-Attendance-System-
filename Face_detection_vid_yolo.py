import cv2
import torch

# Load the trained YOLOv5 model
model = torch.hub.load('D:\\Guna\\TARP_Temp\\YOLOv5_Model', 'custom',
                       path='D:\\Guna\\TARP_Temp\\YOLOv5_Model\\runs\\train\\exp\\weights\\best.pt', force_reload=True,
                       source='local')

# Changing settings to prevent finding the faces multiple times
model.conf = 0.5
model.iou = 0.3

# Access the video file (change the path to your video file)
video_path = "D:\\Guna\\TARP_Temp\\classroom_video.mp4"
cap = cv2.VideoCapture(video_path)

count = 1  # Counter for saving images

while True:
    # Read frames from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Process the results and save detected faces as images
    for pred in results.pred[0]:
        # Extract coordinates and confidence
        x1, y1, x2, y2, conf, class_pred = pred[:6]  # Extract coordinates, confidence, and class

        # Convert coordinates to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Crop the detected face from the frame
        cropped_face = frame[y1:y2, x1:x2]

        # Save the cropped face as an image
        cv2.imwrite(f'people_{count}.jpg', cropped_face)
        count += 1

    # Display the frame with detections using OpenCV
    cv2.imshow('YOLOv5 Face Detection', frame)

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
