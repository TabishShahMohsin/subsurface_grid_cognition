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

def detect_grid_points(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    
    if lines is None:
        return np.array([]), np.array([])
    
    horizontal = []
    vertical = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if abs(angle) < 10 or abs(angle) > 170:
            horizontal.append(line[0])
        elif 80 <= abs(angle) <= 100:
            vertical.append(line[0])
    
    intersections = []
    for h in horizontal:
        for v in vertical:
            x1, y1, x2, y2 = h
            x3, y3, x4, y4 = v
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            if denom == 0:
                continue
            ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
            ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                intersections.append((x, y))
    
    if len(intersections) == 0:
        return np.array([]), np.array([])
    
    intersections = np.array(intersections)
    sorted_pts = sorted(intersections, key=lambda p: (p[1], p[0]))
    rows = []
    current_row = [sorted_pts[0]]
    last_y = sorted_pts[0][1]
    for p in sorted_pts[1:]:
        if abs(p[1] - last_y) < 10:
            current_row.append(p)
        else:
            current_row.sort(key=lambda p: p[0])
            rows.append(current_row)
            current_row = [p]
            last_y = p[1]
    current_row.sort(key=lambda p: p[0])
    rows.append(current_row)
    
    grid_points = []
    obj_points = []
    for i, row in enumerate(rows):
        for j, pt in enumerate(row):
            grid_points.append(pt)
            obj_points.append([j, i, 0.0])
    
    return np.array(grid_points, dtype=np.float32), np.array(obj_points, dtype=np.float32)

def main():
    img1 = cv2.imread('pic1.jpeg')
    img2 = cv2.imread('pic2.jpeg')
    
    if img1 is None or img2 is None:
        print("Error: Could not load images.")
        return
    
    camera_matrix = np.array([[800, 0, img1.shape[1] / 2],
                              [0, 800, img1.shape[0] / 2],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    
    pts1, obj_pts1 = detect_grid_points(img1)
    pts2, obj_pts2 = detect_grid_points(img2)
    
    if len(pts1) < 4 or len(pts2) < 4:
        print("Not enough grid points detected in one or both images.")
        return
    
    success1, rvec1, tvec1 = cv2.solvePnP(obj_pts1, pts1, camera_matrix, dist_coeffs)
    success2, rvec2, tvec2 = cv2.solvePnP(obj_pts2, pts2, camera_matrix, dist_coeffs)
    
    if not success1 or not success2:
        print("Pose estimation failed.")
        return
    
    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)
    
    R_rel = R2 @ R1.T
    t_rel = tvec2 - R_rel @ tvec1
    
    print("Relative Translation (Grid Units):")
    print(f"tx: {t_rel[0][0]:.2f}, ty: {t_rel[1][0]:.2f}, tz: {t_rel[2][0]:.2f}")

if __name__ == "__main__":
    main()