import cv2
import torch

# Load the trained YOLOv5 model
model = torch.hub.load('D:\\Guna\\TARP_Temp\\YOLOv5_Model', 'custom',
                       path='D:\\Guna\\TARP_Temp\\YOLOv5_Model\\runs\\train\\exp\\weights\\best.pt', force_reload=True,
                       source='local')

# Changing settings to prevent finding the faces multiple times
model.conf = 0.5
model.iou = 0.3

# Access the webcam
cap = cv2.VideoCapture(1)  # Change the value to 1 or 2 if multiple cameras are available

while True:
    # Read frames from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Process the results and draw bounding boxes around faces
    for pred in results.pred[0]:
        # Extract coordinates and confidence
        x1, y1, x2, y2, conf, class_pred = pred[:6]  # Extract coordinates, confidence, and class

        # Convert coordinates to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw bounding box rectangle on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Change color and thickness if needed

    # Display the frame with detections using OpenCV
    cv2.imshow('YOLOv5 Face Detection', frame)

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
