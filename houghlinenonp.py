import cv2
import numpy as np
from itertools import combinations

def draw_infinite_line(image, rho, theta, color=(0, 255, 0), thickness=2):
    """Draw a line represented by rho and theta on the image."""
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    # Points far in both directions to simulate infinite line
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return (x1, y1, x2, y2)

def compute_intersection(line1, line2):
    """Find intersection point of two lines given as (x1,y1,x2,y2)."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Line 1 coefficients
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1

    # Line 2 coefficients
    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    det = A1 * B2 - A2 * B1
    if det == 0:
        return None  # parallel lines

    x = (B2 * C1 - B1 * C2) / det
    y = (A1 * C2 - A2 * C1) / det
    return int(x), int(y)

# Load and preprocess image
image = cv2.imread("19.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Apply Standard Hough Transform
lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

segments = []  # Store drawn endpoints

if lines is not None:
    for line in lines:
        rho, theta = line[0]
        pts = draw_infinite_line(image, rho, theta)
        segments.append(pts)

    # Find and draw intersection points
    for l1, l2 in combinations(segments, 2):
        pt = compute_intersection(l1, l2)
        if pt:
            x, y = pt
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:  # Within image bounds
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

# Show or save result
cv2.imshow("Standard Hough Lines with Intersections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()