import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

noise_img = cv.imread("./images/gaussian Noise.png", cv.IMREAD_UNCHANGED)

## Image filtering using 2D convolution

# [1] Apply custom averaging filter (Lowpass Filter) using filter2D()
kernal = np.ones((5, 5), np.float32)/25
dst = cv.filter2D(noise_img, -1, kernal)
result1 = np.hstack((noise_img, dst))
plt.subplot(3, 2, 1), plt.imshow(cv.cvtColor(result1, cv.COLOR_BGR2RGB))


# [2] Apply averaging filter (Lowpass Filter) : (3 * 3 filter)
dst = cv.blur(noise_img, (3, 3))
result2 = np.hstack((noise_img, dst))
plt.subplot(3, 2, 2), plt.imshow(cv.cvtColor(result2, cv.COLOR_BGR2RGB))


# Box filter
# [3] Apply averaging filter (Lowpass Filter) : (3 * 3 box filter)
dst = cv.boxFilter(noise_img, -1, (3, 3))
result3 = np.hstack((noise_img, dst))
plt.subplot(3, 2, 3), plt.imshow(cv.cvtColor(result3, cv.COLOR_BGR2RGB))


# Median Filtering and Gaussian Filtering
# [4] Apply averaging filter : (Median value = 3)
dst = cv.medianBlur(noise_img, 3)
result4 = np.hstack((noise_img, dst))
plt.subplot(3, 2, 4), plt.imshow(cv.cvtColor(result4, cv.COLOR_BGR2RGB))


# Gaussian Filtering
# [5] Apply averaging filter : (Median value = 3)
dst = cv.GaussianBlur(noise_img, (11, 11), 0)
result5 = np.hstack((noise_img, dst))
plt.subplot(3, 2, 5), plt.imshow(cv.cvtColor(result5, cv.COLOR_BGR2RGB))


plt.show()