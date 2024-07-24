import cv2
import numpy as np
img1 = cv2.imread("C:\\Users\\KIIT\\Desktop\\Project\\Images_Coloured\\jug.jpg")
img1=cv2.resize(img1,(640,360))
gray_image = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original Image', img1)
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey()   #OR cv.waitKey(0) both wait indefinitely
cv2.destroyAllWindows