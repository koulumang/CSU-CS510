import cv2 as cv

img = cv.imread('phone.jpg', cv.IMREAD_GRAYSCALE)
img = cv.imread('1.png', cv.IMREAD_GRAYSCALE)
img = cv.imread('9.png', cv.IMREAD_GRAYSCALE)

edges = cv.Canny(img,100,200)
cv.imshow('edges', edges)
cv.waitKey(0)
