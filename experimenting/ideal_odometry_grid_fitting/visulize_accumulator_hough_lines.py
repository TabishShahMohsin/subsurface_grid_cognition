from skimage.transform import hough_line
from skimage.feature import canny
from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np

import cv2


image = io.imread('a.png')
print(image.shape)
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# image = color.rgb2gray(io.imread('original.jpeg'))
edges = canny(image)
h, theta, d = hough_line(edges)

plt.imshow(np.log(1 + h), extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
           cmap='hot', aspect='auto')
plt.title('Hough accumulator')
plt.xlabel('Angles (degrees)')
plt.ylabel('Distance (pixels)')
plt.colorbar()
plt.show()