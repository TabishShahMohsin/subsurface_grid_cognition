import numpy as np
import cv2
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def draw_projected_grid(h, w, pose, spacing=0.1, K=None, return_grid=False):
    # Auto extent based on camera height (tz)
    rx, ry, rz, tx, ty, tz = pose
    if K is None:
        K = np.array([[800, 0, w / 2], [0, 800, h / 2], [0, 0, 1]])

    fov = 2 * np.arctan(max(h, w) / (2 * K[0, 0]))
    extent = int(((tz + 0.1) * np.tan(fov / 2)) / spacing) + 2

    coords = []
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            coords.append([j * spacing, i * spacing, 0])
    coords = np.array(coords, dtype=np.float32)

    Rx = cv2.Rodrigues(np.array([rx, 0, 0]))[0]
    Ry = cv2.Rodrigues(np.array([0, ry, 0]))[0]
    Rz = cv2.Rodrigues(np.array([0, 0, rz]))[0]
    R = Rz @ Ry @ Rx
    t = np.array([[tx], [ty], [tz]])
    RT = np.hstack((R, t))
    P = K @ RT

    coords_h = np.hstack((coords, np.ones((coords.shape[0], 1))))
    proj = (P @ coords_h.T).T
    proj_2d = proj[:, :2] / proj[:, 2:]

    if return_grid:
        return proj_2d
    else:
        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        grid = proj_2d.reshape((2 * extent + 1, 2 * extent + 1, 2))
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1] - 1):
                pt1 = tuple(np.int32(grid[i, j]))
                pt2 = tuple(np.int32(grid[i, j + 1]))
                cv2.line(img, pt1, pt2, (0, 0, 0), 1)
        for j in range(grid.shape[1]):
            for i in range(grid.shape[0] - 1):
                pt1 = tuple(np.int32(grid[i, j]))
                pt2 = tuple(np.int32(grid[i + 1, j]))
                cv2.line(img, pt1, pt2, (0, 0, 0), 1)
        return img


def optimize_grid_to_image(image, detected_points):
    h, w = image.shape[:2]
    K = np.array([[800, 0, w / 2],
                  [0, 800, h / 2],
                  [0,   0,     1]])

    # === Loss function: distance between projected grid and detected corners
    def loss(pose):
        grid_proj = draw_projected_grid(h, w, pose, K=K, return_grid=True)
        distances = []
        for dp in detected_points:
            dists = np.linalg.norm(grid_proj - dp, axis=1)
            distances.append(np.min(dists))  # take closest grid point to this corner
        return np.mean(distances)

    # === Initial random pose (rx, ry, rz, tx, ty, tz)
    # initial_pose = (0.001, 0.001, 0.001, 0.001, 0.001, 0.001)
    initial_pose = np.random.uniform(low=[-0.2, -0.2, -0.2, -1, -1, 1],
                                     high=[0.2, 0.2, 0.2, 1, 1, 3])

    # === Optimize using Powell (or Nelder-Mead if unstable)
    result = minimize(loss, initial_pose, method='Powell', options={'maxiter': 300})
    best_pose = result.x

    # === Final overlay
    final_grid = draw_projected_grid(h, w, best_pose, K=K)
    cv2.imshow("Optimized Grid on Image", final_grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return best_pose

    
# Load your image and detected corners
img = cv2.imread("../samples/shutter_stock/a.png")  # or whatever your input is
corners = np.array([[100, 200], [150, 200], [200, 200], [100, 250], [150, 250]])  # example points

pose = optimize_grid_to_image(img, corners)
print("Optimized Pose:", pose)