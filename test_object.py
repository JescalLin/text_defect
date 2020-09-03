import cv2
import numpy as np

img = cv2.imread('img/all.jpg')


gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
kernel_size = 7
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
low_threshold = 100
high_threshold = 180
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)



kernel = np.ones((60,60),np.uint8)  
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

(cnts, _) = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    if h>200 and w >1500:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 35)
        box_w = int(w/5)
        cv2.rectangle(img, (x,y), (x+box_w, y+h), (0, 0, 255), 15)
        cv2.rectangle(img, (x+box_w,y), (x+2*box_w, y+h), (0, 0, 255), 15)
        cv2.rectangle(img, (x+2*box_w,y), (x+3*box_w, y+h), (0, 0, 255), 15)
        cv2.rectangle(img, (x+3*box_w,y), (x+4*box_w, y+h), (0, 0, 255), 15)
        cv2.rectangle(img, (x+4*box_w,y), (x+5*box_w, y+h), (0, 0, 255), 15)




cv2.imshow('img', cv2.resize(img,(600,400)))
cv2.imshow('closing', cv2.resize(closing,(600,400)))

cv2.waitKey(0)
