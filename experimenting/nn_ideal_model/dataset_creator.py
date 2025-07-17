import cv2
import numpy as np
import os
import csv
from tqdm import tqdm

def rotation_matrix(roll, pitch, yaw):
    roll, pitch, yaw = np.radians([roll, pitch, yaw])
    Rx = np.array([[1, 0, 0], 
                   [0, np.cos(roll), -np.sin(roll)], 
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], 
                   [0, 1, 0], 
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], 
                   [np.sin(yaw), np.cos(yaw), 0], 
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

def project_points(points_3d, R, t, K):
    Rt = R.T
    t_inv = -Rt @ t
    points_cam = (Rt @ points_3d.T).T + t_inv
    mask = points_cam[:, 2] > 0
    points_proj = (K @ points_cam.T).T
    points_2d = points_proj[:, :2] / points_proj[:, 2:3]
    return points_2d.astype(np.int32), mask

def generate_grid_image(roll, pitch, yaw):
    # Camera parameters (fixed)
    fx = fy = 800
    cx = cy = 400
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    t = np.array([0, 0, -200])  # Camera position
    # GPT messed up conventions so minus had to be added
    
    # Generate rotation matrix
    R = rotation_matrix(roll, pitch, yaw)
    
    # Grid parameters (30x30 tiles)
    tile_w, tile_h = 5, 3  # cm
    rows, cols = 30, 30
    grid_cx = cols * tile_w / 2
    grid_cy = rows * tile_h / 2
    
    # Create blank image
    img = np.ones((800, 800, 3), dtype=np.uint8) * 255
    
    # Draw grid
    for i in range(rows):
        for j in range(cols):
            x0 = j * tile_w - grid_cx
            y0 = i * tile_h - grid_cy
            corners = np.array([
                [x0, y0, 0],
                [x0 + tile_w, y0, 0],
                [x0 + tile_w, y0 + tile_h, 0],
                [x0, y0 + tile_h, 0]
            ])
            pts_2d, mask = project_points(corners, R, t, K)
            if mask.all():
                cv2.polylines(img, [pts_2d], True, (0, 0, 0), 1)
    return img

def generate_dataset(num_samples, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'labels.csv')
    
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['filename', 'roll', 'pitch', 'yaw'])
        
        for i in tqdm(range(num_samples), desc="Generating dataset"):
            # Generate random angles (±10 degrees)
            roll = np.random.uniform(-10, 10) 
            pitch = np.random.uniform(-10, 10) 
            yaw = np.random.uniform(-10, 10) 
            
            # Generate image
            img = generate_grid_image(roll, pitch, yaw)
            
            # Save image and label
            filename = f'image_{i:05d}.png'
            img_path = os.path.join(output_dir, filename)
            cv2.imwrite(img_path, img)
            csv_writer.writerow([filename, roll, pitch, yaw])

# Generate 10,000 samples (adjust as needed)
generate_dataset(num_samples=1000, output_dir='grid_dataset')