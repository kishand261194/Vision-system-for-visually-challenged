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




def yellow_f(A):
    box = [[0 for i in range(4)] for j in range(1)]
    hsv = cv2.cvtColor(A, cv2.COLOR_BGR2HSV)
    
    lower_yellow = np.array([20,55,55])
    upper_yellow = np.array([40,255,255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
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
        cv2.rectangle(A,(box[0][0],box[0][1]),(box[0][0]+box[0][2],box[0][1]+box[0][3]),(0,0,0),2)
        cv2.putText(A,'yellow',(box[0][0],box[0][1]), font, 1,(0,255,255),2)
        crop = A[box[0][1]:box[0][1]+box[0][3],box[0][0]:box[0][0]+box[0][2]]
        train.ext(crop,0)



def green_f(A):
    box = [[0 for i in range(4)] for j in range(1)]
    hsv = cv2.cvtColor(A, cv2.COLOR_BGR2HSV)
    
    lower_green = np.array([40,55,55])
    upper_green = np.array([60,255,255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
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
        cv2.putText(A,'green',(box[0][0],box[0][1]), font, 1,(0,255,0),2)
        crop = A[box[0][1]:box[0][1]+box[0][3],box[0][0]:box[0][0]+box[0][2]]
        train.ext(crop,1)
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
        train.ext(crop,2)


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
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # define range of blue color in HSV
    lower_blue = np.array([110,55,55])
    upper_blue = np.array([130,255,255])

    
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    
    #cv2.imshow('frame',frame)
    #cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    
 
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
    if (max_area < 120000) :
        s1=s1+1
        if(s1==8):
            # system('say Come closer')
            s1=0
        print "Come closer"
    if(max_area>100000 and max_area< 170000 ):
        s2=s2+1
        if(s2==25):
            system('say perfect')
            s2=0
        print "Perfect"
    if(max_area > 170000):
        s3=s3+1
        if(s3==8):
            # system('say too close')
            s3=0
        print "Too close"

#   print max_area

#   cv2.drawContours(res,contours,ci,(0,255,0),-1)
#    cv2.imshow("frame2",res)
    x=0
    y=0
    w=0
    h=0
    if(ci==0 or ci<0 or max_area<40000 ):
        print "Object not found try looking around"
        continue
    else:
        print "Object found stay still"
        cnt=contours[ci]
        x,y,w,h = cv2.boundingRect(cnt)
        
        
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),6)        #boundry
        
        
        
        cv2.rectangle(frame,(x,y+int(h/3)),(x+w,y+int(2*h/3)),(0,0,0),2) #3
        
        cv2.rectangle(frame,(x,y),(x+w,y+int(h/3)),(0,0,0),2)     #1
        
        
        cv2.rectangle(frame,(x,y+int(2*h/3)),(x+w,y+h),(0,0,0),2) #5
        
        A = frame[y:y+int(h/3),x:x+w]
        B = frame[y+int(h/3):y+int(h*2/3),x:x+w]
        C = frame[y+int(2*h/3):y+h,x:x+w]
        
        yellow_f(A)
        yellow_f(B)
        yellow_f(C)
        green_f(A)
        green_f(B)
        green_f(C)
        red_f(A)
        red_f(B)
        red_f(C)
        cv2.imshow("final",frame)



#       fun(A,B,C)
#        cv2.imshow("test",A)
#        cv2.imshow("test1",B)
#        cv2.imshow("test2",C)
#        cv2.imshow("test3",D)
#        cv2.imshow("test4",E)
#        cv2.imshow("test5",F)




    if y<50 and (y+h)<380:
        print "look up"
    elif y>100 and (y+h)>450:
        print "look down"
    if x<100 and (x+w)<550:
        print "look left"
    elif x>150 and (x+w)>580:
        print "look right"
#################################################################################################





    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()













