import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from skimage.transform import hough_line, hough_line_peaks
import math
from final import *
from back_froth import *


data = pd.read_csv('/home/tanminh/Documents/Uong/caominh/output/frame_2.csv')
P = data[['X', 'Y']].values.astype(np.float32)
theta = 15
theta_rad = np.deg2rad(theta)
R_theory = np.array([[math.cos(theta_rad), -math.sin(theta_rad)],
                    [math.sin(theta_rad), math.cos(theta_rad)]])
t_theory = np.array([5, 5])
Q = np.dot(R_theory, P.T).T + t_theory.T
df_transformed = pd.DataFrame(Q, columns=['X', 'Y'])
output_file_path = 'transformed_points.csv'
df_transformed.to_csv(output_file_path, index=False)

#####
image = create_image_from_points(data)
lines = detect_lines_with_hough(image)
output_file_path = 'detected_endpoints.csv'
endpoints_df = extract_and_save_endpoints(lines, data, output_file_path)
intersections_df = find_intersections(lines, data, angle_threshold=60)
#print(intersections_df)
data_2 = pd.read_csv('transformed_points.csv')
image_2 = create_image_from_points(data_2)
lines_2 = detect_lines_with_hough(image_2)
output_file_path_2 = '/home/tanminh/Documents/Uong/caominh/final/detected_endpoints_2.csv'
endpoints_df_2 = extract_and_save_endpoints(lines_2, data_2, output_file_path)
intersections_df_2 = find_intersections(lines_2, data_2, angle_threshold=60)




trans, R, T = icp(intersections_df,intersections_df_2 )



# Visualization with intersections
plt.figure(figsize=(10, 8))

plt.scatter(data['X'], data['Y'], color='gray', alpha=0.5, s=100, label='Point Cloud')
plt.scatter(endpoints_df['X'], endpoints_df['Y'], color='crimson', s=100, label='Endpoints')
plt.scatter(intersections_df['X'], intersections_df['Y'], color='green', s=100, label='Intersections')



plt.scatter(data_2['X'], data_2['Y'], color='blue', alpha=0.5, s=100, label='Point Cloud2')
plt.scatter(endpoints_df_2['X'], endpoints_df_2['Y'], color='yellow', s=100, label='Endpoints2')
plt.scatter(intersections_df_2['X'], intersections_df_2['Y'], color='black', s=100, label='Intersections2')

plt.title('Point Cloud with Detected Lines, Endpoints, and Intersections')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.show()