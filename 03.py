import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('./images/color.png', )


# plotting using opencv
hist1 = cv.calcHist([img], [0], None, [256], [0, 256])
hist2 = cv.calcHist([img], [1], None, [256], [0, 256])
hist3 = cv.calcHist([img], [2], None, [256], [0, 256])

plt.plot(hist1, color='r')
plt.plot(hist2, color='g')
plt.plot(hist3, color='b')

plt.show()


# plotting using numpy
# hist, bins = np.histogram(img.ravel(), 256, [0, 256])
# plt.plot(hist)
# plt.show()