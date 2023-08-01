import cv2
import numpy as np
import os

input_directory = './input_images/'
output_directory = './output_images/'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.where((255- v) < value, 255, v + value)
    final_hsv = cv2.merge((h, s, v))
    brighter_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return brighter_img

for filename in os.listdir(input_directory):
    if filename.endswith(".jpg"): 
        img = cv2.imread(input_directory + filename)

        # Calculate the center square coordinates for cropping
        h, w = img.shape[:2]
        start_row, start_col = int(h * .3), int(w * .3)
        end_row, end_col = int(h * .6), int(w * .6)

        # Crop the image
        cropped_img = img[start_row:end_row, start_col:end_col]

        # Increase the brightness of the cropped image
        brighter_img = increase_brightness(cropped_img, value=30)

        # Convert the brighter image to grayscale
        gray = cv2.cvtColor(brighter_img, cv2.COLOR_BGR2GRAY)

        # Set a brightness threshold level
        brightness_threshold = 210  # adjust this value based on your image's brightness

        # Apply the threshold
        _, thresholded = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)

        # Save the thresholded image
        base = os.path.splitext(filename)[0]
        cv2.imwrite(f"{output_directory}{base}_2.jpg", thresholded)
