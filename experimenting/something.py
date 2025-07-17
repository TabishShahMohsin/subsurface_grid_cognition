import cv2
import numpy as np

def detect_corners(image, max_corners=1000):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=10)
    if corners is not None:
        corners = np.squeeze(corners)
    return corners

def match_to_grid(corners, tile_size=0.1, grid_size=(5, 5)):
    """
    Mock-up: Return a simulated world grid to match the detected corners.
    In a real app, this should be matched using spatial sorting.
    """
    rows, cols = grid_size
    obj_points = []
    for i in range(rows):
        for j in range(cols):
            obj_points.append([j * tile_size, i * tile_size, 0])
    return np.array(obj_points[:len(corners)], dtype=np.float32)

def estimate_pose(img, img_points, obj_points):
    K = np.array([[800, 0, img.shape[1] / 2],
                  [0, 800, img.shape[0] / 2],
                  [0,   0, 1]], dtype=np.float64)
    dist = np.zeros((5, 1))  # assume no lens distortion
    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist)
    if not success:
        raise ValueError("Could not solve PnP.")
    rot_mat, _ = cv2.Rodrigues(rvec)
    angles = cv2.decomposeProjectionMatrix(np.hstack((rot_mat, tvec)))[-1].flatten()
    return angles  # pitch, yaw, roll

def main(image_path):
    img = cv2.imread(image_path)

    corners = detect_corners(img)
    if corners is None or len(corners) < 4:
        raise ValueError("Not enough corners detected.")

    obj_points = match_to_grid(corners)
    img_points = np.array(corners[:len(obj_points)], dtype=np.float32)

    pitch, yaw, roll = estimate_pose(img, img_points, obj_points)
    print(f"\nEstimated Camera Angles (degrees):")
    print(f"  Pitch (X-axis tilt): {pitch:.2f}°")
    print(f"  Yaw   (Z-axis turn): {yaw:.2f}°")
    print(f"  Roll  (Y-axis tilt): {roll:.2f}°")

    # Optional visualization
    for pt in img_points:
        cv2.circle(img, tuple(np.int32(pt)), 4, (0, 0, 255), -1)
    cv2.imshow("Detected Corners", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run this function with an image path
if __name__ == "__main__":
    import sys
    image_path = sys.argv[-1]  # e.g., "tiles.jpg"
    main(image_path)