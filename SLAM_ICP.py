import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from skimage.transform import hough_line, hough_line_peaks
import math
import os
import re
from final import *
from back_froth import *
import imageio.v2 as imageio

input_directory_path = '/home/tanminh/Documents/Uong/caominh/output'
output_directory_path = '/home/tanminh/Documents/Uong/caominh/final/image'

def process_2frame(path_file_frame1, path_file_frame2):


    path1 = os.path.join(input_directory_path, path_file_frame1) 
    path2 = os.path.join(input_directory_path, path_file_frame2)  
    frame1 = pd.read_csv(path1)
    frame2 = pd.read_csv(path2)

    data1 = frame1[['X', 'Y']].values.astype(np.float32)
    data2 = frame2[['X', 'Y']].values.astype(np.float32)
    print(data1)

    image1 = create_image_from_points(frame1)
    lines1 = detect_lines_with_hough(image1)
    intersections_frame1 = find_intersections(lines1, frame1, angle_threshold=60)

    image2 = create_image_from_points(frame2)
    lines2 = detect_lines_with_hough(image2)
    intersections_frame2 = find_intersections(lines2, frame2, angle_threshold=60)

    frame_trans, R, T = icp(intersections_frame1,intersections_frame2)

    return R,T 




Position = [[0, 0]]
"""
def read_frames_from_directory(directory):
    frames = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(directory, filename))

            frames.append(df.values)

    return frames

"""
def extract_number(filename):
    # Sử dụng regex để tìm tất cả các số trong tên file và lấy số đầu tiên
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    return 0 

def compute_robot_path(frames):
    path = [(0, 0)]
    current_position = np.array([0, 0])

    images = []
    gif_filename = os.path.join(output_directory_path, 'slam_path.gif')

    for i in range(1, len(frames)):
        previous_frame = frames[i-1]
        current_frame = frames[i]

        R, t = process_2frame(previous_frame, current_frame)

        current_position = np.dot(R, current_position) + t
        path.append((current_position[0], current_position[1]))

        plt.figure()
        plt.plot([p[0] for p in path], [p[1] for p in path], marker='o', linestyle='-')
        plt.scatter(current_position[0], current_position[1], color='red')
        
        #current_frame_data = pd.read_csv(current_frame)[['X', 'Y']].values.astype(np.float32)
        #plt.scatter(current_frame_data[:, 0], current_frame_data[:, 1], color='gray', s=1)
        
        plt.title('SLAM Path of the Robot')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid()

        temp_filename = f'temp_frame_{i}.png'
        plt.savefig(temp_filename)
        plt.close()
        
        images.append(imageio.imread(temp_filename))

    imageio.mimsave(gif_filename, images, duration=0.5)
    for temp_filename in [f'temp_frame_{i}.png' for i in range(1, len(frames))]:
        os.remove(temp_filename)

    plt.show()

frames = [file for file in os.listdir(input_directory_path) if file.endswith('.csv')]
frames.sort(key=extract_number)

compute_robot_path(frames)









