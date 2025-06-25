import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("sample.png", cv2.IMREAD_COLOR)

print(img[255, 255])    # Direct pixel values
print(type(img))    # Numpy array

# Code for creating a white bar on the image
for i in range(10):
    img[55 + i,] = [255, 255, 255]
    
# Region of image: ROI
roi = img[100:150, 100:150] = [255, 255, 255]

watch_face = img[37:111, 107:194]
img[0:74, 0:87] = watch_face

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()