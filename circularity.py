import cv2
import numpy as np
import os
import csv

input_directory = './output_images/'
output_directory = './output/'
output_csv = './circularity.csv'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def connect_the_dots(points, epsilon_factor=0.04):
    hull = cv2.convexHull(points)
    epsilon = epsilon_factor * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    return approx

def calculate_circularity(points):
    perimeter = cv2.arcLength(points, True)
    area = cv2.contourArea(points)
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    return circularity

for filename in os.listdir(input_directory):
    if filename.endswith(".jpg"):
        img = cv2.imread(input_directory + filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold the image to obtain binary image of white dots
        _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours of white dots
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Combine all the dot points into a single array
        dots = np.concatenate(contours)

        # Connect the dots to form a larger circle with adjustable epsilon
        epsilon_factor = 0.001  # Adjust this value for smoother or more degenerate edges
        connected_dots = connect_the_dots(dots, epsilon_factor)

        # Calculate the circularity of the larger circle
        circularity = calculate_circularity(connected_dots)

        # Draw the connected dots and the circularity on the image
        result = img.copy()
        cv2.drawContours(result, [connected_dots], -1, (0, 255, 0), 2)
        text = f"Circularity: {circularity:.2f}"
        cv2.putText(result, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save the result image
        base = os.path.splitext(filename)[0]
        cv2.imwrite(f"{output_directory}{base}_result.jpg", result)

        # Save circularity to csv
        with open(output_csv, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([filename, circularity])
