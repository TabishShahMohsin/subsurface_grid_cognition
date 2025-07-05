import cv2
import numpy as np
from scipy.optimize import minimize
# IN chaining in video loop we would need to use the prev values to init the next values


def rotation_matrix(roll, pitch, yaw):
    # Correction for looking down
    Rc = np.array([
            [1, 0,0],
            [0, -1, 0],
            [0, 0, -1],
    ])
    roll, pitch, yaw = np.radians([roll, pitch, yaw])
    Rx = np.array(
        [[1, 0, 0],
         [0, np.cos(roll), -np.sin(roll)],
         [0, np.sin(roll), np.cos(roll)]
    ])
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
    return Rz @ Ry @ Rx @ Rc


def project_points(points_3d, R, t, K):
    # Inverse transform: world to camera coordinates
    Rt = R.T
    t_inv = -Rt @ t
    points_cam = (Rt @ points_3d.T).T + t_inv
    mask = points_cam[:, 2] > 0  # keep only points in front of the camera
    points_proj = (K @ points_cam.T).T
    points_2d = points_proj[:, :2] / points_proj[:, 2:3]
    return points_2d.astype(np.int32), mask


def draw_scene(roll, pitch, yaw, tx, ty, tz):
    # Camera intrinsics
    fx = fy = 800
    cx = cy = 400
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Rotation and translation of the camera
    R = rotation_matrix(roll, pitch, yaw)
    t = np.array([tx, ty, tz])

    # Grid setup
    tile_w, tile_h = 5, 3  # cm
    rows, cols = 25, 15 
    grid_cx = cols * tile_w / 2
    grid_cy = rows * tile_h / 2

    img = np.zeros((800, 800), dtype=np.uint8) * 255

    for i in range(rows):
        for j in range(cols):
            # Grid centered at origin
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
                cv2.polylines(img, [pts_2d], True, (255, 255, 255), 1)

    return img

def loss_function(pose, img2: np.array) -> float:    
    roll, pitch, yaw, tx, ty, tz = pose
    img1 = draw_scene(roll, pitch, yaw, tx, ty, tz)
    cv2.imshow('something', img1)
    cv2.waitKey(0)
    cv2.imshow('something', img2)
    cv2.waitKey(0)
    abs_diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    l1_loss = np.mean(abs_diff)
    return l1_loss


roll, pitch, yaw, tx, ty, tz = 0, 0, 0, 0, 0, 50
img = cv2.imread('sample.png')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = cv2.Canny(img, 100, 200)

init_pose = np.array([0, 0, 0, 0, 0, 20]) 
print(loss_function(init_pose, img))

bounds = [
        (-90, 90),    # roll
        (-90, 90),    # pitch
        (-180, 180),  # yaw
        (-200, 200),  # tx
        (-200, 200),  # ty
        (10, 500)     # tz
    ]

res = minimize(
    loss_function,
    init_pose,
    args=(img, ),
    method='L-BFGS-B',
    bounds=bounds,
    options={'maxiter':300, 'disp' : True,}
)

optimized_pose = res.x
print(optimized_pose)