import cv2
import numpy as np
from numpy import sqrt,arccos,rad2deg
from os import system
from extract import Extract
import time
from veri import veri
import speech_recognition as sr


##################################
import csv
from sklearn.naive_bayes import GaussianNB
import sklearn
tester=Extract()
csv = np.genfromtxt ('target1.csv', delimiter=",")
target = csv[:,0]

y=[int(i)for i in target]
Y=np.array(y)
import csv
with open('data1.csv', 'rb') as f:
    reader = csv.reader(f)
    lst = list(reader)
    x = [[(float(j)) for j in i] for i in lst]
    x = [np.array(i) for i in x]
X=np.array(x)
clf = GaussianNB()
clf.fit(X,Y)
###################################
font = cv2.FONT_HERSHEY_SIMPLEX
novo=0

class cor:
    hx=0
    hy=0
    hw=0
    hh=0
    x=0
    y=0
    w=0
    h=0
    hmidx=0
    hmidy=0
    midx=0
    midy=0
    f=0
    st=0

t=cor()

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
    if (max_area > 1500 ) :
        cnt=contours[ci]
        box[0][0],box[0][1],box[0][2],box[0][3] = cv2.boundingRect(cnt)
        t.x,t.y,t.w,t.h=cv2.boundingRect(cnt)
        t.midx=(t.x+t.x+t.w)/2
        t.midy=(t.y+t.y+t.h)/2
        cv2.rectangle(A,(box[0][0],box[0][1]),(box[0][0]+box[0][2],box[0][1]+box[0][3]),(0,0,0),2)
        cv2.putText(A,'Sugar',(box[0][0],box[0][1]), font, 1,(0,255,255),2)
        crop = A[box[0][1]:box[0][1]+box[0][3],box[0][0]:box[0][0]+box[0][2]]
    else:
        system('say Your hand is parallel to the object, try moving your hand forward')
        time.sleep(3)
        system('say Hope you picked the container ,  yes or no ? ')
        h=1
        while(h==1):

# obtain audio from the microphone
            r = sr.Recognizer()
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source)

                print "say"
                audio = r.listen(source)

# recognize speech using Sphinx

            if (r.recognize_sphinx(audio)=="yes"):
                system("say  Great job, can you please look at the picked object")
                veri(3)
            elif(r.recognize_sphinx(audio)=="no"):
                system("say Dont worry ,lets do it again")
                h=0
            elif(r.recognize_sphinx(audio)!="no" and r.recognize_sphinx(audio)!="yes") :
                system("say -v vicki Please speak again")
        novo=1



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
    if (max_area > 1500 ) :
        cnt=contours[ci]
        box[0][0],box[0][1],box[0][2],box[0][3] = cv2.boundingRect(cnt)
        t.x,t.y,t.w,t.h=cv2.boundingRect(cnt)
        t.midx=(t.x+t.x+t.w)/2
        t.midy=(t.y+t.y+t.h)/2
        cv2.rectangle(A,(box[0][0],box[0][1]),(box[0][0]+box[0][2],box[0][1]+box[0][3]),(255,255,255),2)
        cv2.putText(A,'Coffee powder',(box[0][0],box[0][1]), font, 1,(0,255,0),2)
    else:

        system('say Your hand is parallel to the object, try moving your hand forward')
        time.sleep(3)
        system('say Hope you picked the container ,  yes or no ? ')
        h=1
        while(h==1):
    # obtain audio from the microphone
            r = sr.Recognizer()
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source)

                print("Say something!")
                audio = r.listen(source)

            # recognize speech using Sphinx

            if (r.recognize_sphinx(audio)=="yes"):
                system("say  Great job, can you please look at the picked object")
                veri(2)
            elif(r.recognize_sphinx(audio)=="no"):
                system("say  Dont worry ,lets do it again")
                h=0
            elif(r.recognize_sphinx(audio)!="no" and r.recognize_sphinx(audio)!="yes") :
                system("say  Please speak again")
        novo=1




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
    c12=0
    max_area2=100
    z=0
    x=0
    for i in range(len(contours)):
        cnt=contours[i]
        area = cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci=i

    if (max_area > 1500 ) :
        cnt=contours[ci]
        box[0][0],box[0][1],box[0][2],box[0][3] = cv2.boundingRect(cnt)
        t.x,t.y,t.w,t.h=cv2.boundingRect(cnt)
        crop = A[box[0][1]:box[0][1]+box[0][3],box[0][0]:box[0][0]+box[0][2]]
        ans=tester.test(crop)
        ans=clf.predict(ans)
        print ans
        if(ans==1):
            z=1
            cv2.rectangle(A,(box[0][0],box[0][1]),(box[0][0]+box[0][2],box[0][1]+box[0][3]),(255,255,255),2)
            cv2.putText(A,'Milk',(box[0][0],box[0][1]), font, 1,(0,0,255),2)
            t.midx=(t.x+t.x+t.w)/2
            t.midy=(t.y+t.y+t.h)/2




    for i in range(len(contours)):
        cnt=contours[i]
        area = cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci=i
        elif max_area > area > max_area2:
            max_area2 = area
            ci2=i

    if (max_area2 > 1500 ) :
        cnt=contours[ci2]
        box[0][0],box[0][1],box[0][2],box[0][3] = cv2.boundingRect(cnt)
        t.x,t.y,t.w,t.h=cv2.boundingRect(cnt)
        crop = A[box[0][1]:box[0][1]+box[0][3],box[0][0]:box[0][0]+box[0][2]]
        ans=tester.test(crop)
        ans=clf.predict(ans)
        print ans
        if(ans==1 and z!=1):
            x=1
            cv2.rectangle(A,(box[0][0],box[0][1]),(box[0][0]+box[0][2],box[0][1]+box[0][3]),(255,255,255),2)
            cv2.putText(A,'Milk',(box[0][0],box[0][1]), font, 1,(0,0,255),2)
            t.midx=(t.x+t.x+t.w)/2
            t.midy=(t.y+t.y+t.h)/2





    if(z==0 and x==0):

        system('say Your hand is parallel to the object, try moving your hand forward')
        time.sleep(3)
        system('say Hope you picked the container ,  yes or no ? ')
        while(h==1):
            # obtain audio from the microphone
            r = sr.Recognizer()
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source)

                print("Say something!")
                audio = r.listen(source)

            # recognize speech using Sphinx

                if (r.recognize_sphinx(audio)=="yes"):
                    system("say Great job, can you please look at the picked object")
                    veri(1)
                elif(r.recognize_sphinx(audio)=="no"):
                    system("say Dont worry ,lets do it again")
                    h=0
                elif(r.recognize_sphinx(audio)!="no" and r.recognize_sphinx(audio)!="yes") :
                    system("say Please speak again")
        novo=1










def hand(n):
    t.f=0
    cap = cv2.VideoCapture(1)
    box = [[0 for i in range(4)] for j in range(1)]
    while( cap.isOpened() ) :
        novo=0
        ret,img = cap.read()
        OriginalImg = img.copy()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        ret,thresh1 = cv2.threshold(blur,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        MIN = np.array([0,48,80],np.uint8)
        MAX = np.array([20,255,255],np.uint8) #HSV: V-79%
        HSVImg = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        res = cv2.inRange(HSVImg,MIN,MAX)
        res = cv2.erode(res,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        res = cv2.dilate(res,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        cv2.imshow('res',res)
        if(t.st==0):
            system("say -v vicki     Please extend your hand forward")
            t.st=1
        if(n==1):
            f=green_f(OriginalImg)

        #return f
        elif(n==2):
            f=red_f(OriginalImg)

        #  return f
        elif(n==3):
            f =yellow_f(OriginalImg)
            
        #    return f


        contours, hierarchy = cv2.findContours(res,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
#    tempImage = np.zeros(img.shape,np.uint8)

        if len(contours)==0:
            continue

        Index = []
        index_val = 0

   # for cnt in contours:
        tempImage = img.copy()
        tempImage = cv2.subtract(tempImage,img)
        max_area=0
        ci=0
        for i in range(len(contours)):
            cnt=contours[i]
            area = cv2.contourArea(cnt)
            if(area>max_area):
                max_area=area
                ci=i
        cnt=contours[ci]

        if ci==0:
            continue

        if (max_area > 5000) :
            cnt=contours[ci]
            box[0][0],box[0][1],box[0][2],box[0][3] = cv2.boundingRect(cnt)
            t.hx,t.hy,t.hw,t.hh=cv2.boundingRect(cnt)

            crop1 = OriginalImg[box[0][1]:box[0][1]+box[0][3],box[0][0]:box[0][0]+box[0][2]]
            ans=tester.test(crop1)
            ans=clf.predict(ans)
            if ans==0:
                cv2.rectangle(OriginalImg,(box[0][0],box[0][1]),(box[0][0]+box[0][2],box[0][1]+box[0][3]),(0,0,0),2)
                cv2.putText(OriginalImg,'hand',(box[0][0],box[0][1]), font, 1,(0,0,255),2)
                t.hmidx=(t.hx+(t.hx+t.hw))/2
                t.hmidy=(t.hy+(t.hy+t.hy))/2

                if novo==0:
                    if(t.hmidx-50<t.midx):
                        system("say  lil right")
                    elif(t.hmidx+50>t.midx):
                        system("say  lil left")
                    if(t.hmidy+50<t.midy):
                        system("say lil down")
                    if(t.hmidy-50>t.midy):
                        system("say lil up")






        cv2.imshow("Finger tracking",OriginalImg)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
