import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from skimage.transform import hough_line, hough_line_peaks
import math

# Load the data again


# Create an image from point cloud data
def create_image_from_points(data, image_size=(1000, 1000)):
    # Normalize points
    x_normalized = (data['X'] - data['X'].min()) / (data['X'].max() - data['X'].min())
    y_normalized = (data['Y'] - data['Y'].min()) / (data['Y'].max() - data['Y'].min())
    
    # Scale to image size
    x_scaled = (x_normalized * (image_size[0] - 1)).astype(int)
    y_scaled = (y_normalized * (image_size[1] - 1)).astype(int)
    
    # Create an image with white background
    image = np.zeros(image_size, dtype=np.uint8)
    image[y_scaled, x_scaled] = 255  # Set points to white
    
    return image

# Generate the image from the point data


# Function to detect lines using Hough transform (adapted to use OpenCV)
def detect_lines_with_hough(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
    return lines

# Detect lines in the generated image


# Visualization of the generated image and the detected lines
'''
plt.figure(figsize=(10, 6))
plt.imshow(image, cmap='gray', origin='lower')
if lines is not None:
    for line in lines:
        for x1, y1, x2, y2 in line:
            plt.plot([x1, x2], [y1, y2], 'r')
plt.title('Detected Lines in Image')
plt.show()
'''


def extract_and_save_endpoints(lines, data, output_file_path):
    endpoints = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Convert image coordinates to data coordinates
            x1_data = (x1 / 1000) * (data['X'].max() - data['X'].min()) + data['X'].min()
            y1_data = (y1 / 1000) * (data['Y'].max() - data['Y'].min()) + data['Y'].min()
            x2_data = (x2 / 1000) * (data['X'].max() - data['X'].min()) + data['X'].min()
            y2_data = (y2 / 1000) * (data['Y'].max() - data['Y'].min()) + data['Y'].min()
            endpoints.append([x1_data, y1_data])
            endpoints.append([x2_data, y2_data])
    
    # Convert list to DataFrame
    endpoint_df = pd.DataFrame(endpoints, columns=['X', 'Y'])
    
    # Save to CSV file
    endpoint_df.to_csv(output_file_path, index=False)
    return endpoint_df

# Save the endpoints to a CSV file

'''
def line_parameters(x1, y1, x2, y2):
    # Edge case: vertical line to avoid division by zero
    if x2 - x1 == 0:
        return float('inf'), x1  # Slope is infinity and x1 is the x-intercept for vertical lines
    else:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return slope, intercept

# Helper function to calculate the intersection of two lines
def calculate_intersection(slope1, intercept1, slope2, intercept2):
    if slope1 == slope2:
        return None  # Parallel lines or identical lines
    if slope1 == float('inf'):  # Line 1 is vertical
        x = intercept1
        y = slope2 * x + intercept2
    elif slope2 == float('inf'):  # Line 2 is vertical
        x = intercept2
        y = slope1 * x + intercept1
    else:
        x = (intercept2 - intercept1) / (slope1 - slope2)
        y = slope1 * x + intercept1
    return x, y

# Helper function to calculate the angle between two lines
def angle_between_lines(slope1, slope2):
    if slope1 == float('inf'):
        slope1 = 1e10  # A very large number to approximate infinity slope
    if slope2 == float('inf'):
        slope2 = 1e10  # A very large number to approximate infinity slope
    tan_theta = abs((slope2 - slope1) / (1 + slope1 * slope2))
    angle_radians = math.atan(tan_theta)
    return math.degrees(angle_radians)
'''

def find_intersections(lines, data, angle_threshold=60):
    def line_parameters(x1, y1, x2, y2):
        if x2 - x1 == 0:  # Vertical line edge case
            return float('inf'), x1
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return slope, intercept

    def calculate_intersection(slope1, intercept1, slope2, intercept2):
        if slope1 == slope2:
            return None  # Parallel or identical lines
        if slope1 == float('inf'):  # Line 1 is vertical
            x = intercept1
            y = slope2 * x + intercept2
        elif slope2 == float('inf'):  # Line 2 is vertical
            x = intercept2
            y = slope1 * x + intercept1
        else:
            x = (intercept2 - intercept1) / (slope1 - slope2)
            y = slope1 * x + intercept1
        return x, y

    def angle_between_lines(slope1, slope2):
        if slope1 == float('inf') or slope2 == float('inf'):
            tan_theta = 1.0  # Approximating for very steep slopes
        else:
            tan_theta = abs((slope2 - slope1) / (1 + slope1 * slope2))
        return math.degrees(math.atan(tan_theta))

    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            x1, y1, x2, y2 = lines[i][0]
            x3, y3, x4, y4 = lines[j][0]
            slope1, intercept1 = line_parameters(x1, y1, x2, y2)
            slope2, intercept2 = line_parameters(x3, y3, x4, y4)
            angle = angle_between_lines(slope1, slope2)
            if angle > angle_threshold:
                intersection = calculate_intersection(slope1, intercept1, slope2, intercept2)
                if intersection:
                    intersections.append(intersection)

    # Mapping intersections back to original scale
    intersections_df = pd.DataFrame(intersections, columns=['X', 'Y'])
    intersections_df['X'] = (intersections_df['X'] / 1000) * (data['X'].max() - data['X'].min()) + data['X'].min()
    intersections_df['Y'] = (intersections_df['Y'] / 1000) * (data['Y'].max() - data['Y'].min()) + data['Y'].min()

    return intersections_df
# Detect intersections with angle constraints
'''
intersections = []
for i in range(len(lines)):
    for j in range(i + 1, len(lines)):
        x1, y1, x2, y2 = lines[i][0]
        x3, y3, x4, y4 = lines[j][0]
        slope1, intercept1 = line_parameters(x1, y1, x2, y2)
        slope2, intercept2 = line_parameters(x3, y3, x4, y4)
        angle = angle_between_lines(slope1, slope2)
        if angle > 60:  # Only consider intersections where the angle > 60 degrees
            intersection = calculate_intersection(slope1, intercept1, slope2, intercept2)
            if intersection:
                intersections.append(intersection)
print(intersections)
# Convert intersections for plotting
intersections_df = pd.DataFrame(intersections, columns=['X', 'Y'])
intersections_df['X'] = (intersections_df['X'] / 1000) * (data['X'].max() - data['X'].min()) + data['X'].min()
intersections_df['Y'] = (intersections_df['Y'] / 1000) * (data['Y'].max() - data['Y'].min()) + data['Y'].min()
'''
