import cv2
import numpy as np
import os

# Load your image (replace 'your_image.jpg' with the actual file path)
image_path = r'C:\Users\sruja\OneDrive\Desktop\ALL FILES\Flask\Images\Cheque 309123.jpg'
image = cv2.imread(image_path)

# Grayscale operation
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gaussian filtering
sigma = 1  # Adjust the value of sigma based on your requirement
gaussian_filtered = cv2.GaussianBlur(gray_image, (0, 0), sigma)

# Binary image conversion
_, binary_image = cv2.threshold(gaussian_filtered, 136, 255, cv2.THRESH_BINARY)

# Create the 'grayimg' folder if it doesn't exist
output_folder = 'grayimg'
os.makedirs(output_folder, exist_ok=True)

# Save the results in the 'grayimg' folder
cv2.imwrite(os.path.join(output_folder, 'gray_image.jpg'), gray_image)
cv2.imwrite(os.path.join(output_folder, 'gaussian_filtered_image.jpg'), gaussian_filtered)
cv2.imwrite(os.path.join(output_folder, 'binary_image.jpg'), binary_image)

# Display or save the results as needed
# cv2.imshow('Original Image', image)
# cv2.imshow('Grayscale Image', gray_image)
# cv2.imshow('Gaussian Filtered Image', gaussian_filtered)
# cv2.imshow('Binary Image', binary_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
