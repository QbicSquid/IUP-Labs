import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


bright_img = cv.imread('./images/brightImage.png', 0)
dark_img = cv.imread('./images/darkImage.png', 0)
color_img = cv.imread('./images/color.png', 1)

### Histogram Equalization to improve the contrast of a dark and bright image
# Create histogram of the image
dark_img_hist = cv.calcHist([dark_img], [0], None, [256], [0, 256])
bright_img_hist = cv.calcHist([bright_img], [0], None, [256], [0, 256])
# Apply histogram equalization
equalized_dark_img = cv.equalizeHist(dark_img)
equalized_bright_img = cv.equalizeHist(bright_img)
# Create histogram of the equalized image
equalized_dark_img_hist = cv.calcHist([equalized_dark_img], [0], None, [256], [0, 256])
equalized_bright_img_hist = cv.calcHist([equalized_bright_img], [0], None, [256], [0, 256])


### Color Histogram Equalization to improve the contrast of color image
# Convert image from RGB to HSV
color_img_hsv = cv.cvtColor(color_img, cv.COLOR_RGB2HSV)
# Histogram equalization on V channel
color_img_hsv[:,:,2] = cv.equalizeHist(color_img_hsv[:,:,2])
# Convert image from HSV to RGB
color_img_rgb = cv.cvtColor(color_img_hsv, cv.COLOR_HSV2RGB)
# Create histogram of the image
color_img_hist = cv.calcHist([color_img], [0], None, [256], [0, 256])
color_img_rgb_hist = cv.calcHist([color_img_rgb], [0], None, [256], [0, 256])


# Plot the dark_img_hist & equalized_dark_img_hist
# : plt.subplot(211) - 1st parameter value means Row, 2nd parameter value means column, 3rd parameter value means image no
plt.subplot(3, 2, 1), plt.plot(dark_img_hist)
plt.subplot(3, 2, 2), plt.plot(equalized_dark_img_hist)
plt.subplot(3, 2, 3), plt.plot(bright_img_hist)
plt.subplot(3, 2, 4), plt.plot(equalized_bright_img_hist)
plt.subplot(3, 2, 5), plt.plot(color_img_hist)
plt.subplot(3, 2, 6), plt.plot(color_img_rgb_hist)

plt.show()