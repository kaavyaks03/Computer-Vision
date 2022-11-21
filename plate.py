import os
import re
import cv2 #open cv library
import numpy as np
import imutils
import pytesseract
from os.path import isfile, join
import matplotlib.pyplot as plt

#read the image
image = cv2.imread('archive/Numberplate/40.jpg')
cv2.imshow("Original",image)
cv2.waitKey(0)

#convert to gray scale
img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayed image",img_grey) 
cv2.waitKey(0)

#noise reduction
img=cv2.bilateralFilter(img_grey,11,17,17)
cv2.imshow("Smoothened image",img)
cv2.waitKey(0)

#edge detection using canny filter
edged = cv2.Canny(img,30,200)
cv2.imshow("Edged image",edged)
cv2.waitKey(0)

#find contours
#set a thresh
thresh = 100
#get threshold image
ret,thresh_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
#find contours
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


#create an empty image for contours
img_contours = np.zeros(image.shape)
# draw the contours on the empty image
cv2.drawContours(img_contours, contours, -1, (0,255,0), 3)

cv2.imshow("Contours",img_contours)
cv2.waitKey(0)

#select rectangle number plate from contours
cnt = sorted(cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
mask = np.zeros((256,256), np.uint8)
masked = cv2.drawContours(mask, [cnt],-1, 255, -1)

plt.axis('off')
cv2.imshow(masked)

#masking the image
dst = cv2.bitwise_and(image, image, mask=mask)
segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

#writing it as a seperate image
cv2.imwrite('/Car vdo dataset/Figure 7.jpg')
