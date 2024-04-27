import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

## Negative transformation
img_pre = cv.imread('./images/apples.jpg', cv.IMREAD_UNCHANGED)
img = cv.cvtColor(img_pre, cv.COLOR_BGR2RGB) # only needed because apples.jpg is in BGR

plt.subplot(3, 1, 1), plt.imshow(img_pre)
plt.subplot(3, 1, 2), plt.imshow(img)

for i in img:
  for j in i:
    j[0] = 255 - j[0]
    j[1] = 255 - j[1]
    j[2] = 255 - j[2]

plt.subplot(3, 1, 3), plt.imshow(img)
plt.show()


## Power-law transformation to improve contrast
img = cv.imread('./images/gamma.jpg', cv.IMREAD_UNCHANGED)

gamma_2_2 = np.array(255*(img/255)**2.2, dtype='uint8')
gamma_0_4 = np.array(255*(img/255)**0.4, dtype='uint8')

plt.subplot(1, 3, 1), plt.imshow(img)
plt.subplot(1, 3, 2), plt.imshow(gamma_2_2)
plt.subplot(1, 3, 3), plt.imshow(gamma_0_4)
plt.show()


## Log transformation to improve the dynamic range of an image
img = cv.imread('./images/log.jpg', cv.IMREAD_GRAYSCALE)

log_img = (np.log(img + 1 ) / (np.log( 1 + np.max( img )))) * 255
np.seterr(divide='ignore')
log_img = np.array(log_img, dtype= np.uint8)

plt.subplot(1, 2, 1), plt.imshow(img, cmap="gray")
plt.subplot(1, 2, 2), plt.imshow(log_img, cmap="gray")
plt.show()