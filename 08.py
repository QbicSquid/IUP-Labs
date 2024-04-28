import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('./images./building.jpg', 0)

# [1] Apply Laplacian Filter - 8bits not enough for that
# So we use cv.CV_64F for 64bits(Convertian Flag)
laplacian = cv.Laplacian(img, cv.CV_64F)
# Scaling the output - (Ignore negative values)
laplacianAbs = cv.convertScaleAbs(laplacian)
# Apply laplacian gaussian
laplacianAbsGaussian = cv.GaussianBlur(laplacianAbs, (5,5), cv.BORDER_DEFAULT)

# Display the result
plt.subplot(2, 2, 1), plt.imshow(img)
plt.subplot(2, 2, 2), plt.imshow(laplacian)
plt.subplot(2, 2, 3), plt.imshow(laplacianAbs)
plt.subplot(2, 2, 4), plt.imshow(laplacianAbsGaussian)
plt.show()


# [2] Apply Sobel Operator
sobelX = cv.Sobel(img, cv.CV_64F, 1, 0, 3)
sobelY = cv.Sobel(img, cv.CV_64F, 0, 1, 3)

# Scaling the output - (Ignore negative values)
sobelXAbs = cv.convertScaleAbs(sobelX)
sobelYAbs = cv.convertScaleAbs(sobelY)

# Display the result
plt.subplot(2, 2, 1), plt.imshow(sobelX)
plt.subplot(2, 2, 2), plt.imshow(sobelY)
plt.subplot(2, 2, 3), plt.imshow(sobelXAbs)
plt.subplot(2, 2, 4), plt.imshow(sobelYAbs)
plt.show()