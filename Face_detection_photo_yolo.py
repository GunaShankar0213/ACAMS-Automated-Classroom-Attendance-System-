import cv2
import torch
import os

# Load the trained YOLOv5 model
model = torch.hub.load('D:\\Guna\\TARP_Temp\\YOLOv5_Model', 'custom',
                       path='D:\\Guna\\TARP_Temp\\YOLOv5_Model\\runs\\train\\exp\\weights\\best.pt', force_reload=True,
                       source='local')

# Changing settings to prevent finding the faces multiple times
model.conf = 0.5
model.iou = 0.3

# Path to the folder containing images (change this to your image folder)
image_folder = "D:\\Guna\\TARP_Temp\\photo_extract\\"

count = 1  # Counter for saving images

# Iterate through images in the folder
for image_name in os.listdir(image_folder):
    # Read each image
    image_path = os.path.join(image_folder, image_name)
    frame = cv2.imread(image_path)

    # Perform inference on the image
    results = model(frame)

    # Process the results and save detected faces as images
    for pred in results.pred[0]:
        # Extract coordinates and confidence
        x1, y1, x2, y2, conf, class_pred = pred[:6]  # Extract coordinates, confidence, and class

        # Convert coordinates to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Crop the detected face from the image
        cropped_face = frame[y1:y2, x1:x2]

        # Save the cropped face as an image
        cv2.imwrite(f'D:\\Guna\\TARP_Temp\\photo_extract\\people_{count}.jpg', cropped_face)
        count += 1

    # Display the image with detections using OpenCV (optional)
    cv2.imshow('YOLOv5 Face Detection', frame)
    cv2.waitKey(1000)  # Display the image for 1 second (adjust as needed)

# Close OpenCV window
cv2.destroyAllWindows()
