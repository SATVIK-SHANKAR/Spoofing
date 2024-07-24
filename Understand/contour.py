import cv2 as cv
import numpy as np
img1 = cv.imread("C:\\Users\\KIIT\\Desktop\\Project\\Images_Coloured\\sample.jpg")
img1=cv.resize(img1,(640,360))

gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
cv.imshow('Grayscale Image', gray)
canny=cv.Canny(img1,125,175)
cv.imshow('Canny', canny)
# find contours takes parameter first the edge image and second is the mode used to find contours and third the contour approximation method RETR_LIST means all contours
contours,hierarchies=cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
cv.imshow("Ans",img1)
cv.waitKey()
cv.destroyAllWindows