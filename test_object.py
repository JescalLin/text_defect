import cv2
import numpy as np

cropped_img_Text = []

def process_rgb(i,rgb):
    rgb2 = rgb.copy()
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    morphKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,5))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, morphKernel)
    # binarize
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # connect horizontally oriented regions
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, morphKernel)

    # find contours
    mask = np.zeros(bw.shape[:2], dtype="uint8")
    contours, hierarchy = cv2.findContours(connected, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # filter contours
    idx = 0
    i = 0
    while idx >= 0:
        x,y,w,h = cv2.boundingRect(contours[idx])
        # fill the contour
        cv2.drawContours(mask, contours, idx, (255, 255, 255), cv2.FILLED)
        # ratio of non-zero pixels in the filled region
        

        r = float(cv2.contourArea(contours[idx])/(w*h))
        if(r > 0.45  and h<w and y>200):
            cv2.rectangle(rgb, (x,y), (x+w,y+h), (0, 255, 0), 2)
            if(i<1 and w>250 and h>16):
                cropped_img_Text.append(rgb2[y:y+h, x:w+x])
                i = 1
        idx = hierarchy[0][idx][0]
    return rgb

def process_text(i,text_rgb):
    gray = cv2.cvtColor(text_rgb, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    binary = 255 - binary
    closing1 = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)) )
    closing2 = cv2.morphologyEx(binary,cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) )
    contours_c1, hierarchy = cv2.findContours(closing1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_c2, hierarchy = cv2.findContours(closing2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    
    for c in contours_c1:
        x,y,w,h = cv2.boundingRect(c)
        if (h >65) :
            cv2.rectangle(text_rgb, (x,y), (x+w,y+h), (0, 0, 255), 2)

    for c in contours_c2:
        x,y,w,h = cv2.boundingRect(c)
        if (h<15 and w <15) :
            cv2.rectangle(text_rgb, (x,y), (x+w,y+h), (255, 0, 255), 2)

    
    return text_rgb







"""
img = cv2.imread('img/5.jpg')
num = 5
"""
img = cv2.imread('img/5_2.jpg')
num = 5

img_rgb2 = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
kernel_size = 7
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
low_threshold = 100
high_threshold = 180
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

kernel = np.ones((60,60),np.uint8)  
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


(cnts, _) = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cut_image=[]
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    if h>200 and w >300:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 25)
        box_w = int(w/num)
        for i in range(num):
            cv2.rectangle(img, (x+i*box_w,y), (x+(i+1)*box_w, y+h), (0, 0, 255), 15)
            cut_image.append(img_rgb2[y:y+h, x+i*box_w:x+(i+1)*box_w])


for i in range(num):
    cut_image[i] = process_rgb(i,cut_image[i])
    cv2.imshow(str(i), cv2.resize(cut_image[i],(250,350)))



for i in range(num):
    # process_text 偵測文字瑕疵
    h, w = cropped_img_Text[0].shape[:2]
    cropped_img_Text[i] = process_text(i,cropped_img_Text[i])
    cv2.imshow("cropped_img_Text"+str(i), cropped_img_Text[i])




cv2.imshow('img', cv2.resize(img,(720,320)))

cv2.waitKey(0)
cv2.destroyAllWindows()