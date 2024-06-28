import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_distance_matrix(A, B):
    
    M = A.shape[0]
    N = B.shape[0]
    D = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            D[i, j] = np.sqrt((A[i, 0] - B[j, 0])**2 + (A[i, 1] - B[j, 1])**2)
    return D

def find_minimum_distances(A, B):
    D = calculate_distance_matrix(A, B)
    
    min_dist_A_to_B = np.min(D, axis=1)
    min_indices_A_to_B = np.argmin(D, axis=1)
    
    
    min_dist_B_to_A = np.min(D, axis=0)
    min_indices_B_to_A = np.argmin(D, axis=0)
    
    
    A_coress = []
    B_coress = []
    
    for i in range(len(min_dist_A_to_B)):
        b_index = min_indices_A_to_B[i]
        if min_indices_B_to_A[b_index] == i:
            A_coress.append(A[i])
            B_coress.append(B[b_index])
            
    return np.array(A_coress), np.array(B_coress)

def best_fit_transform(A, B):
    assert A.shape == B.shape

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = np.dot(AA.T, BB)

    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = centroid_B.T - np.dot(R, centroid_A.T)

    return R, t

def icp(A, B, max_iterations=20, tolerance=1e-5):
    #P_transformed = P.copy()
    A = np.array(A)
    B = np.array(B)
    prev_error = float('inf')

    for i in range(max_iterations):
        #correspondences = dtw_2d(A, B)
        #A_corr = np.array([A[i] for i, j in correspondences])
        #B_corr = np.array([B[j] for i, j in correspondences])

        A_corr, B_corr = find_minimum_distances(A, B)

        R, t = best_fit_transform(A_corr, B_corr)

        A_trans = np.dot(R, A.T).T + t.T
        A_trans_corr = np.dot(R, A_corr.T).T + t.T

        mean_error = np.mean(np.linalg.norm(A_trans_corr - B_corr, axis=1))

        if abs(prev_error - mean_error) < tolerance:
            break

        prev_error = mean_error
    theta = np.arctan2(R[1,0],R[0, 0])
    angle = np.rad2deg(theta)
    print("Rotation matrix R:")
    print(R)
    print("Theta angle:")
    print(angle)
    print("Translation vector t:")
    print(t)

    return A_trans, R, t
'''
Path_1 = 'output/frame_2.csv'
Path_2 = 'output/frame_3.csv'
file_path_1 = 'Frame_2_endpoint.csv'
file_path_2 = 'Frame_3_endpoint.csv'

df3 = pd.read_csv(Path_1)
P1 = df3[['X', 'Y']].values.astype(np.float32)
df4 = pd.read_csv(Path_2)
Q1 = df4[['X', 'Y']].values.astype(np.float32)

df1 = pd.read_csv(file_path_1)
P = df1[['X', 'Y']].values.astype(np.float32)
df2 = pd.read_csv(file_path_2)
Q = df2[['X', 'Y']].values.astype(np.float32)


P_trans, R, t = icp(P, Q)

P_trans1, R1, t1 = icp(P1, Q1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(P[:, 0], P[:, 1], color='blue', label='Frame T0')
ax.scatter(Q[:, 0], Q[:, 1], color='red', label='Frame T1 meansure')
ax.scatter(P_trans[:, 0], P_trans[:, 1], color='green', label='Frame T1 estimate')


ax.scatter(P1[:, 0], P1[:, 1], color='orange', label='Frame T01')
ax.scatter(Q1[:, 0], Q1[:, 1], color='cyan', label='Frame T11 meansure')
ax.scatter(P_trans1[:, 0], P_trans1[:, 1], color='gray', label='Frame T11 estimate')
ax.legend()
ax.set_title('ICP Alignment for LIDAR 2D Scans')
ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.show()

'''
