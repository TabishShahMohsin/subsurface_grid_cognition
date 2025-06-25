import cv2
import numpy as np
import math
from sklearn.cluster import KMeans

def compute_vanishing_point(group):
    intersections = []
    for i in range(len(group)):
        for j in range(i+1, len(group)):
            (x1, y1), (x2, y2), _ = group[i]
            (x3, y3), (x4, y4), _ = group[j]

            A = np.array([
                [x2 - x1, x3 - x4],
                [y2 - y1, y3 - y4]
            ])
            b = np.array([x3 - x1, y3 - y1])

            if np.linalg.matrix_rank(A) == 2:
                t = np.linalg.solve(A, b)
                xi = x1 + t[0] * (x2 - x1)
                yi = y1 + t[0] * (y2 - y1)
                intersections.append([xi, yi])

    if not intersections:
        raise ValueError("No valid line intersections found for vanishing point.")

    return np.mean(intersections, axis=0)

def get_camera_pose_from_vanishing_points(image_path, fx, fy, cx, cy):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('1. Grayscale Image', gray)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    cv2.imshow('2. Canny Edges', edges)

    # Detect lines using Probabilistic Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)
    if lines is None:
        raise ValueError("No lines detected")

    # Initialize angle list and group list for clustering
    angles = []
    line_groups = []
    line_img = img.copy()

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.atan2((y2 - y1), (x2 - x1))
        angles.append(angle)
        line_groups.append(((x1, y1), (x2, y2), angle))
        # Draw all detected lines
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.imshow('3. All Detected Lines', line_img)

    # Cluster lines based on angle using KMeans (2 dominant directions)
    X = np.array(angles).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init=10).fit(X)
    labels = kmeans.labels_

    # Separate the lines into two groups
    group1 = [line_groups[i] for i in range(len(line_groups)) if labels[i] == 0]
    group2 = [line_groups[i] for i in range(len(line_groups)) if labels[i] == 1]

    # Draw clustered lines
    clustered_img = img.copy()
    for (x1, y1), (x2, y2), _ in group1:
        cv2.line(clustered_img, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Blue lines
    for (x1, y1), (x2, y2), _ in group2:
        cv2.line(clustered_img, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Red lines
    cv2.imshow('4. Clustered Line Groups (Blue & Red)', clustered_img)

    # Compute vanishing points for each group
    vp1 = compute_vanishing_point(group1)
    vp2 = compute_vanishing_point(group2)
    print(f"The 2 vanishing points found are: {vp1} and {vp2}.")

    # Draw vanishing points on image
    vp_img = img.copy()
    cv2.circle(vp_img, tuple(np.int32(vp1)), 8, (0, 255, 255), -1)
    cv2.circle(vp_img, tuple(np.int32(vp2)), 8, (255, 255, 0), -1)
    cv2.imshow('5. Vanishing Points (Yellow & Cyan)', vp_img)

    # Camera matrix
    K_mat = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]])
    Kinv = np.linalg.inv(K_mat)

    # Convert vanishing points to direction vectors in camera space
    vp1_dir = Kinv @ np.array([vp1[0], vp1[1], 1.0])
    vp2_dir = Kinv @ np.array([vp2[0], vp2[1], 1.0])

    vp1_dir /= np.linalg.norm(vp1_dir)
    vp2_dir /= np.linalg.norm(vp2_dir)

    # Decide X and Y axes directions
    if np.abs(vp1[0] - cx) > np.abs(vp2[0] - cx):
        x_dir = vp1_dir
        y_dir = vp2_dir
    else:
        x_dir = vp2_dir
        y_dir = vp1_dir

    # Z direction is cross product of X and Y
    z_dir = np.cross(x_dir, y_dir)
    z_dir /= np.linalg.norm(z_dir)

    # Rotation matrix from world to camera
    R = np.column_stack((x_dir, y_dir, z_dir))

    # Fake translation (not real depth): assume camera looking at Z=0 from above
    t = -R.T @ np.array([0, 0, 1])

    # Convert rotation matrix to Euler angles
    pitch = math.degrees(math.asin(-R[2, 0]))
    roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))

    # Print pose
    print(f"\nRotation (degrees):\n  Roll:  {roll:.2f}\n  Pitch: {pitch:.2f}\n  Yaw:   {yaw:.2f}")
    print(f"Translation (approximate direction):\n  x: {t[0]:.2f}, y: {t[1]:.2f}, z: {t[2]:.2f}")

    # Wait and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return R, t, roll, pitch, yaw

# Example usage:
fx, fy, cx, cy = 800, 800, 320, 240
image_path = "c.png"
get_camera_pose_from_vanishing_points(image_path, fx, fy, cx, cy)