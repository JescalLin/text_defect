import cv2
import numpy as np


cap = cv2.VideoCapture(0)        #開啟攝像頭
while(1):
    
    ret, img = cap.read()

    num = 1

    img_rgb2 = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel_size = 3
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    cv2.imshow('edges', edges)

    kernel = np.ones((100,100),np.uint8)  
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('closing', closing)

    (cnts, _) = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cut_image=[]
    cropped_img_Text = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if h>1 and w >1:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 3)
            box_w = int(w/num)
            for i in range(num):
                cv2.rectangle(img, (x+i*box_w,y), (x+(i+1)*box_w, y+h), (0, 0, 255), 5)
                cut_image.append(img_rgb2[y:y+h, x+i*box_w:x+(i+1)*box_w])

    
    for i in range(num):
        try:
            cv2.imshow(str(i), cv2.resize(cut_image[i],(250,350)))
        except:
            i=1



    cv2.imshow('img', img)


    if cv2.waitKey(1) & 0xFF == ord('q'):   #如果按下q 就截圖儲存並退出

        break
cap.release()
cv2.destroyAllWindows()