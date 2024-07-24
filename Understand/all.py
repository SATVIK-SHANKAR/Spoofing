import cv2 as cv
import numpy as np
img1 = cv.imread("path to folder")
#! REsize
img1=cv.resize(img1,(640,360),interpolation=cv.INTER_CUBIC)  # when resizing from small to big use _linear or _cubic for better quality
#! Grayscale
gray_image = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
cv.imshow('Original Image', img1)
cv.imshow('Grayscale Image', gray_image)
cv.waitKey()
cv.destroyAllWindows
#!Blur
blur = cv.GaussianBlur(img1,(3,3),cv.BORDER_DEFAULT)
cv.imshow("Blur_Image",blur)
#* To increase blur increase the kernel size say (7,7) it has to be odd
# !Edge Detector
canny = cv.Canny(img1,125,175)
cv.imshow("Edges",canny)

