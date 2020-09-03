import cv2
import numpy as np


img_rgb = cv2.imread('img/2.jpg')
img_rgb2 = img_rgb.copy()
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)


template = cv2.imread('img/temp2.JPG',0)
w, h = template.shape[::-1]

rects = []

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.5
loc = np.where( res >= threshold)



for pt0 in zip(*loc[::-1]):
    i_ymins_b = pt0[1]
    i_ymaxs_b = pt0[1] + h
    i_xmins_b = pt0[0]                                                
    i_xmaxs_b = pt0[0] + w
    rects.append([i_xmins_b,i_ymins_b,w, h])
    
combined_array=np.append(rects,rects,1)
combined_list=combined_array.tolist()
result=cv2.groupRectangles(combined_list,1,0.05)

i = 0 
for (x,y,w,h) in result[0]:
    cv2.rectangle(img_rgb,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.rectangle(img_rgb,(x+w+10,y-35),(x+w+550,y+h+35),(0,0,255),3)
    
    cropped_img = img_rgb2[y-35:y+h+20, x+w+10:x+w+550]
    cropped_img_gray  = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    cropped_img_gray = cv2.GaussianBlur(cropped_img_gray,(3, 3), 0)
    ret,cropped_img_th = cv2.threshold(cropped_img_gray, 0, 255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
    (cnts, _) = cv2.findContours(cropped_img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(cropped_img, cnts, -1, (0, 255, 0), 2)
    for c in cnts:

        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(cropped_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if(w >100):
            cv2.rectangle(cropped_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("cropped_img_th",cropped_img_th)
    cv2.imshow("cropped_img",cropped_img)





cv2.imshow("img_rgb",cv2.resize(img_rgb,(800,600)))
cv2.waitKey(0)