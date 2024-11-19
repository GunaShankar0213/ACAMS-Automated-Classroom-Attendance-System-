import cv2
import os

# Path to the folder containing images
folder_path = 'Photo_train'

# List all files in the folder
file_list = os.listdir(folder_path)

# Resize images and convert to grayscale
width, height = 128, 128

for file_name in file_list:
    # Construct the full path to the image
    image_path = os.path.join(folder_path, file_name)

    # Read the image
    image = cv2.imread(image_path)

    # Resize the image to the specified dimensions
    resized_image = cv2.resize(image, (width, height))

    # Convert image to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Display or save the resized and converted image (optional)
    # cv2.imshow('Resized and Grayscale Image', gray_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # If you want to save the processed image back to disk
    # Ensure the output directory exists
    output_folder = 'Processed_Images'
    os.makedirs(output_folder, exist_ok=True)

    # Construct the output path for the processed image
    output_path = os.path.join(output_folder, file_name)

    # Save the processed image
    cv2.imwrite(output_path, gray_image)
