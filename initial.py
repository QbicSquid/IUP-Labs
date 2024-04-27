import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def drawSquare(image, squareSize):
  height = len(image)
  width = len(image[0])

  # horizontal lines
  for i in range(squareSize, width - squareSize):
    image[squareSize][i] = [0, 0, 0]
    image[height - squareSize][i] = [0, 0, 0]

  # vertical lines
  for i in range(squareSize, height - squareSize):
    image[i][squareSize] = [0, 0, 0]
    image[i][width - squareSize] = [0, 0, 0]
  return image


img1 = cv2.imread('./images/brightImage.png')

for i in range(10, 101, 10):
  img2 = drawSquare(img1.copy(), i)

  result = np.hstack((img1, img2))
  cv2.imshow('Image', result)

  cv2.waitKey(500)
  # time.sleep(100)