# Drawing figures onto images
import numpy as np
import cv2
img = cv2.imread('sample.png', cv2.IMREAD_COLOR)

cv2.line(img, (100, 10), (15, 150), (255, 255, 0), 15)

cv2.rectangle(img, (15, 25), (200, 150),(255, 0, 0), 20)
cv2.circle(img, (200, 200), 55, (0, 0, 255), -1)

pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
cv2.polylines(img, [pts], True, (0, 255, 255), 3)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'Learning OpenCV', (0, 130), font, 1, (200, 255, 255), 2, cv2.LINE_AA)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows