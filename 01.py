import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img1 = cv.imread('./images/brightImage.png')
img2 = cv.imread('./images/color.png', )
img3 = cv.imread('./images/darkImage.png')


# display multiple images at once
fig,ax = plt.subplots(1, 3, figsize = (15, 15))
ax[0].imshow(img1)
ax[0].axis('on')
ax[1].imshow(img2)
ax[1].axis('off')
ax[2].imshow(img3)
plt.show()


# enlargen image
plt.figure(figsize=(10, 10))
plt.imshow(img1)
plt.show()


# access pixel value of image
print("shape: ")
print(img2.shape)
print("size: ")
print(img2.size)
print("dtype: ")
print(img2.dtype)
print(img2[50, 39])