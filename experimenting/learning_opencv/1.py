import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("sample.png", 0)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.imshow(img, cmap='grey', interpolation = 'bicubic')
# plt.plot([50, 100], [80, 100], 'r', linewidth=5)
# plt.show()

cv2.imwrite('watchgray.png', img)