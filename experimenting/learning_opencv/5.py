import numpy as np
import cv2

img_1 = cv2.imread('sample.png')
cv2.imshow('image', img_1)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_2 = cv2.flip(img_1, 0)
cv2.imshow('mirrored_image', img_2)
cv2.waitKey(0)
cv2.destroyAllWindows()

add = cv2.add(img_1, img_2) # 255 is if exceeds
cv2.imshow('cv2.add(), exceeding at 255', add)
cv2.waitKey(0)
cv2.destroyAllWindows()

add = img_1 + img_2 # Counter restarts for overflow
cv2.imshow('using +, counter restarts', add)
cv2.waitKey(0)
cv2.destroyAllWindows()

add = img_1 // 2 + img_2 // 2
cv2.imshow('modulo followed by adding, taking avg of images', add)
cv2.waitKey(0)
cv2.destroyAllWindows()

add = cv2.addWeighted(img_1, 0.6, img_2, 0.4)
cv2.imshow('Using addWeighted()', add)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(add.shape)
c = (150, 150)
print(img_1[c], img_2[c], add[c])