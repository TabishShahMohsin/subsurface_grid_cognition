import cv2
import numpy as np
import math
from sklearn.cluster import KMeans

def conventions_not_for_now(angle1, angle2):
    """
    Returns the smallest angular difference between two angles,
    considering them modulo 180 degrees.
    Result is always in [0, 90].
    """
    diff = abs(angle1 - angle2) % 180
    return min(diff, 180 - diff)

def compute_vanishing_point(group):
    intersections = []
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            (x1, y1), (x2, y2), _ = group[i]
            (x3, y3), (x4, y4), _ = group[j]

            A = np.array(
                [[x2 - x1, x3 - x4], [y2 - y1, y3 - y4]]
            )
            b = np.array([x3 - x1, y3 - y1])

            if np.linalg.matrix_rank(A) == 2:
                t = np.linalg.solve(A, b)
                xi = x1 + t[0] * (x2 - x1)
                yi = y1 + t[0] * (y2 - y1)
                intersections.append([xi, yi])

    if not intersections:
        raise ValueError("No valid line intersections found for vanishing point.")

    return np.mean(intersections, axis=0)


def rotation_matrix(roll, pitch, yaw):
    # Correction
    # roll, pitch, yaw = pitch, -(roll - 180), -yaw
    # roll, pitch, yaw = pitch, roll, yaw

    roll, pitch, yaw = np.radians([roll, pitch, yaw])
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )
    Ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    Rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    return Rz @ Ry @ Rx


def project_points(points_3d, R, t, K):
    Rt = R.T
    t_inv = -Rt @ t
    points_cam = (Rt @ points_3d.T).T + t_inv
    mask = points_cam[:, 2] > 0
    points_proj = (K @ points_cam.T).T
    points_2d = points_proj[:, :2] / points_proj[:, 2:3]
    return points_2d.astype(np.int32), mask


def draw_scene(roll, pitch, yaw, tx, ty, tz, path='generated.png'):
    fx = fy = 800
    cx = cy = 400
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    R = rotation_matrix(roll, pitch, yaw)
    t = np.array([tx, ty, tz])

    tile_w, tile_h = 5, 3
    rows, cols = 8, 8
    grid_cx = cols * tile_w / 2
    grid_cy = rows * tile_h / 2

    img = np.ones((800, 800, 3), dtype=np.uint8) * 255

    for i in range(rows):
        for j in range(cols):
            x0 = j * tile_w - grid_cx
            y0 = i * tile_h - grid_cy
            corners = np.array(
                [
                    [x0, y0, 0],
                    [x0 + tile_w, y0, 0],
                    [x0 + tile_w, y0 + tile_h, 0],
                    [x0, y0 + tile_h, 0],
                ]
            )
            pts_2d, mask = project_points(corners, R, t, K)
            if mask.all():
                cv2.polylines(img, [pts_2d], True, (0, 0, 0), 1)

    origin_2d, visible = project_points(np.array([[0, 0, 0]]), R, t, K)
    if visible[0]:
        cv2.circle(img, tuple(origin_2d[0]), 5, (0, 0, 255), -1)
        cv2.putText(
            img,
            "Origin",
            tuple(origin_2d[0] + [10, -10]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    cv2.imwrite(path, img)
    return path


def get_camera_pose_from_vanishing_points(image_path, fx, fy, cx, cy):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('1. Grayscale Image', gray)

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    cv2.imshow('2. Canny Edges', edges)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)
    if lines is None:
        raise ValueError("No lines detected")

    angles = []
    line_groups = []
    line_img = img.copy()

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.atan2((y2 - y1), (x2 - x1))
        angles.append(angle)
        line_groups.append(((x1, y1), (x2, y2), angle))
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.imshow('3. All Detected Lines', line_img)

    X = np.array(angles).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init=10).fit(X)
    labels = kmeans.labels_

    group1 = [line_groups[i] for i in range(len(line_groups)) if labels[i] == 0]
    group2 = [line_groups[i] for i in range(len(line_groups)) if labels[i] == 1]

    clustered_img = img.copy()
    for (x1, y1), (x2, y2), _ in group1:
        cv2.line(clustered_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    for (x1, y1), (x2, y2), _ in group2:
        cv2.line(clustered_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.imshow('4. Clustered Line Groups (Blue & Red)', clustered_img)

    vp1 = compute_vanishing_point(group1)
    vp2 = compute_vanishing_point(group2)

    vp_img = img.copy()
    cv2.circle(vp_img, tuple(np.int32(vp1)), 8, (0, 255, 255), -1)
    cv2.circle(vp_img, tuple(np.int32(vp2)), 8, (255, 255, 0), -1)
    cv2.imshow('5. Vanishing Points (Yellow & Cyan)', vp_img)

    K_mat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    Kinv = np.linalg.inv(K_mat)

    vp1_dir = Kinv @ np.array([vp1[0], vp1[1], 1.0])
    vp2_dir = Kinv @ np.array([vp2[0], vp2[1], 1.0])

    vp1_dir /= np.linalg.norm(vp1_dir)
    vp2_dir /= np.linalg.norm(vp2_dir)

    if np.abs(vp1[0] - cx) > np.abs(vp2[0] - cx):
        x_dir = vp1_dir
        y_dir = vp2_dir
    else:
        x_dir = vp2_dir
        y_dir = vp1_dir

    z_dir = np.cross(x_dir, y_dir)
    z_dir /= np.linalg.norm(z_dir)

    R = np.column_stack((x_dir, y_dir, z_dir))
    t = -R.T @ np.array([0, 0, 1])

    pitch = math.degrees(math.asin(-R[2, 0]))
    roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))

    print(f"\nEstimated Pose from Image:\n  Roll:  {roll:.2f}\n  Pitch: {pitch:.2f}\n  Yaw:   {yaw:.2f}")
    print(f"  Translation Vector: x: {t[0]:.2f}, y: {t[1]:.2f}, z: {t[2]:.2f}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return R, t, roll, pitch, yaw


# -------- Pipeline Entry --------
if __name__ == '__main__':
    # Ground truth pose (set any value you like to simulate camera view)
    roll, pitch, yaw = -3, -174, -5
    tx, ty, tz = 0, 0, 150

    fx = fy = 800
    cx = cy = 400

    img_path = draw_scene(roll, pitch, yaw, tx, ty, tz, path='generated.png')
    _, _, r, p, y = get_camera_pose_from_vanishing_points(img_path, fx, fy, cx, cy)

    roll_error = conventions_not_for_now(roll, r)
    pitch_error = conventions_not_for_now(pitch, p)
    yaw_error = conventions_not_for_now(yaw, y)

    print(roll_error, pitch_error, yaw_error)
