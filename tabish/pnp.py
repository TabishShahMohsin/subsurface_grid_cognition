import cv2
import numpy as np
import itertools

def get_intersections(lines):
    def line_params(x1, y1, x2, y2):
        A = y2 - y1
        B = x1 - x2
        C = A * x1 + B * y1
        return A, B, C

    intersections = []
    for l1, l2 in itertools.combinations(lines, 2):
        x1, y1, x2, y2 = l1[0]
        A1, B1, C1 = line_params(x1, y1, x2, y2)
        x3, y3, x4, y4 = l2[0]
        A2, B2, C2 = line_params(x3, y3, x4, y4)

        determinant = A1 * B2 - A2 * B1
        if determinant == 0:
            continue  # Parallel lines
        x = (C1 * B2 - C2 * B1) / determinant
        y = (A1 * C2 - A2 * C1) / determinant
        intersections.append((int(x), int(y)))
    return intersections

def draw_grid(image, intersections, radius=4):
    for (x, y) in intersections:
        cv2.circle(image, (x, y), radius, (0, 0, 255), -1)
    return image

def process_tile_grid(image_path):
    if image_path.split('.')[-1] == "mp4":
        cap = cv2.VideoCapture(image_path)
        res, frame = cap.read()
        img = frame
    else:
        img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    h, w = img.shape[:2]

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Blurred
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    cv2.imshow("Blurred", blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Gaus
    gaus = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1)
    cv2.imshow("gaus", gaus)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Canny edges
    edges = cv2.Canny(blur, 5, 30, apertureSize=3)
    # edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.imshow("Canny Edges", edges_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=100, maxLineGap=10)

    # Prepare white background
    white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255

    if lines is None:
        print("No lines detected.")
        return img

    # Draw lines on white background
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(white_bg, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.imshow("Hough Lines on White", white_bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Draw intersections on top
    intersections = get_intersections(lines)
    final_img = draw_grid(white_bg.copy(), intersections)
    cv2.imshow("Final Grid Intersections", final_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# === Run it ===
if __name__ == "__main__":
    import sys
    path = "samples/shutter_stock/" + sys.argv[-1]
    process_tile_grid(path)