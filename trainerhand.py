import cv2
import numpy as np
from os import system
from PIL import Image
from Extraction import Extract


train = Extract()
cap = cv2.VideoCapture(1)
font = cv2.FONT_HERSHEY_SIMPLEX

#30-40 yellow
#40-50 green



def red_f(A):
    box = [[0 for i in range(4)] for j in range(1)]
    hsv = cv2.cvtColor(A, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([160,55,55])
    upper_red = np.array([170,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    
    lower_red = np.array([0,48,80])
    upper_red = np.array([20,255,255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    
    mask = cv2.add(mask1,mask2)
    
    
    ret,thresh = cv2.threshold(mask,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    max_area=100
    ci=0
    for i in range(len(contours)):
        cnt=contours[i]
        area = cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci=i
    if (max_area > 1500) :
        cnt=contours[ci]
        box[0][0],box[0][1],box[0][2],box[0][3] = cv2.boundingRect(cnt)
        cv2.rectangle(A,(box[0][0],box[0][1]),(box[0][0]+box[0][2],box[0][1]+box[0][3]),(255,255,255),2)
        cv2.putText(A,'Red',(box[0][0],box[0][1]), font, 1,(0,0,255),2)
        
        #############
        crop = A[box[0][1]:box[0][1]+box[0][3],box[0][0]:box[0][0]+box[0][2]]
        train.ext(crop,1)


s1=0
s2=0
s3=0
nc=0
tes=0

while(1):
    
    #e=pyttsx.init()
    box = [[0 for i in range(4)] for j in range(1)]
    # Take each frame
    _, frame = cap.read()
    
    red_f(frame)
    
    cv2.imshow("final",frame)
    
    
    
    
    
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()













