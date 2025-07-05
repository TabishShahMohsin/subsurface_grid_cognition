# We are going to start with 2 images and take one
    # Extract lines from it
    # Make lines equation using HOUGH lines
    # Extracting grid out of the lines
    # Doing the same for to 2 scenes and hence finding the delta tx, ty, tz out of the grid between the scenes
    # We are aiming so that we would not need to track 2 points but instead we are solving what the grid is and hence not using any grid tracking ML

# The points in the pic are: 
    # X_cam = R_inv(X_world - t)
    # X_proj = K X_cam
    # homogeneous_to_xy(X_proj)

import cv2
import numpy as np
from sklearn.cluster import KMeans
import math

def main():
    # For now just trying to use 2 images, later would need to write pipeline for video/camera-feed
    img1 = cv2.imread('model.py')
    img2 = cv2.imread('changed.jpeg')

    # Checking if the images are loaded
    if img1 is None or img2 is None:
        print("Error: Could not load images.")
        return
    
    # Building the same camera matrix
    fx = fy = 800
    cx = cy = 400
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])

    detect_grid_points(img1)
    detect_grid_points(img2)

    
def detect_grid_points(image):
    # Applying the basic operations for cleaning the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5,), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    # Finding lines using hough lines, should remember that the houghlines uses quantized values.
    lines = cv2.HoughLines(edges, 0.1, np.pi / 180, threshold=300)

    # Would need to separate the 2 parallel set of lines using theta

    group = cluster(lines)
    print(group)

    theta1 = min(group.keys())
    theta2 = max(group.keys())

    c1 = min(group[theta1])
    c2 = min(group[theta2])

    rho1 = group[theta1][1] - group[theta1][0]
    rho2 = group[theta2][1] - group[theta2][0]

    print(c1, c2, rho1, rho2, theta1, theta2)

            
    show_hough_lines(gray, lines)
    ...

    
    
def display(image, win="Not Mentioned"):
    cv2.imshow(win, image)
    cv2.waitKey(0)


def show_hough_lines(image, lines):
    display(image)
    image = image.copy()
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 800 * (-b))
        y1 = int(y0 + 800 * (a))
        x2 = int(x0 - 800 * (-b))
        y2 = int(y0 - 800 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    display(image)
    

import numpy as np
from sklearn.cluster import KMeans
import math

def cluster(hough_lines, n_clusters=2, angle_thresh=10):
    """
    Groups Hough lines by theta (angle) using KMeans and returns a dict:
    {mean_theta: np.array of rhos}, only including lines within angle_thresh degrees of cluster center.

    Args:
        hough_lines (np.ndarray): Hough lines from cv2.HoughLines(), shape (N, 1, 2)
        n_clusters (int): Number of theta clusters to form (usually 2 for grids).
        angle_thresh (float): Angle tolerance in degrees for filtering outliers in each cluster.

    Returns:
        dict[float, np.ndarray]: Dict with mean theta (radians) as key and ρ array as value.
    """
    if hough_lines is None or len(hough_lines) == 0:
        return {}

    # Flatten from (N, 1, 2) to (N, 2)
    lines = hough_lines[:, 0, :]  # Each line: (rho, theta)
    thetas_deg = np.degrees(lines[:, 1]).reshape(-1, 1)  # For clustering

    # KMeans clustering on angle
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto').fit(thetas_deg)
    labels = kmeans.labels_
    centers_deg = kmeans.cluster_centers_.flatten()

    # Initialize dict: theta_mean (in radians) → list of rhos
    theta_rho_dict = {}

    for cluster_id in range(n_clusters):
        # Indices of lines in this cluster
        idxs = np.where(labels == cluster_id)[0]
        rhos = []
        theta_vals = []

        for idx in idxs:
            rho, theta_rad = lines[idx]
            theta_deg = thetas_deg[idx][0]
            if abs(theta_deg - centers_deg[cluster_id]) <= angle_thresh:
                rhos.append(rho)
                theta_vals.append(theta_rad)

        if rhos:
            mean_theta = float(np.mean(theta_vals))  # in radians
            theta_rho_dict[mean_theta] = np.array(rhos)

    return theta_rho_dict

if __name__ == "__main__":
    main()
    
    
    
    
    

