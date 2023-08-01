import cv2
import numpy as np
import os
import csv

input_directory = './input_images/'
output_directory = './output_images/'
output_csv = './data.csv'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.where((255- v) < value, 255, v + value)
    final_hsv = cv2.merge((h, s, v))
    brighter_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return brighter_img, value

for filename in os.listdir(input_directory):
    if filename.endswith(".jpg"): 
        img = cv2.imread(input_directory + filename)

        # Increase the brightness of the image
        brighter_img, brightness_value = increase_brightness(img, value=30)

        # Convert the brighter image to grayscale
        gray = cv2.cvtColor(brighter_img, cv2.COLOR_BGR2GRAY)

        # Set a brightness threshold level
        brightness_threshold = 210  # adjust this value based on your image's brightness

        # Apply the threshold
        _, thresholded = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)

        # Save the thresholded image
        base = os.path.splitext(filename)[0]
        cv2.imwrite(f"{output_directory}{base}_2.jpg", thresholded)

        # Save data to CSV
        with open(output_csv, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([filename, brightness_value])
