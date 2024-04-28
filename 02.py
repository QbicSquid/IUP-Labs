import cv2 as cv
import numpy as np

# Load the video
video = cv.VideoCapture("D:/UNI_FILEs/YEAR_4/Y4S2/IUP/Week 2 - 13 February - 19 February/Lab_02/small_laptop_connections.mov")

# Read frame by frame
while{video.isOpened()}:
    
    ret, frame = video.read()
    
    # Convert GBR to HSV color space
    videoHSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    # Define  the upper range and the lowe range of the blue color
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    
    # Threshold the HSV image from the range of blue color
    maskVideo = cv.inRange(videoHSV, lower_blue, upper_blue)
    
    # Extract the blue colour object alone
    result = cv.bitwise_and(frame, frame, mask = maskVideo)
    
    # Resize the video
    originalVideo = cv.resize(frame, (400, 400))
    cv.imshow('OriginalVideo', originalVideo)
    
    # Resize the video
    maskVideo = cv.resize(maskVideo, (400, 400))
    cv.imshow('maskVideo', maskVideo)
    
    # Resize the video
    finalVideo = cv.resize(result, (400, 400))
    cv.imshow('finalVideo', finalVideo)
    
    if cv.waitKey(1) == ord('q'):
        break
    
video.release()
cv.destroyAllWindows()