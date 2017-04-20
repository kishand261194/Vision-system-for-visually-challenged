import cv2
import numpy as np
from os import system
from hand import hand
from PIL import Image
import pytesseract
import csv
from sklearn.naive_bayes import GaussianNB
import sklearn
from Extraction import Extract
from audiooo import sp



class var:
    sugar = 0
    coffee = 0
    milk = 0

e=var()
tester=Extract()

csv = np.genfromtxt ('target.csv', delimiter=",")
target = csv[:,0]

y=[int(i)for i in target]
Y=np.array(y)
import csv
with open('data.csv', 'rb') as f:
    reader = csv.reader(f)
    lst = list(reader)
    x = [[(float(j)) for j in i] for i in lst]
    x = [np.array(i) for i in x]
X=np.array(x)
clf = GaussianNB()
clf.fit(X,Y)


cap = cv2.VideoCapture(1)
font = cv2.FONT_HERSHEY_SIMPLEX

#30-40 yellow
#40-50 green




def yellow_f(A,n):
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
        cv2.putText(A,'Sugar',(box[0][0],box[0][1]), font, 1,(0,255,255),2)
        crop = A[box[0][1]:box[0][1]+box[0][3],box[0][0]:box[0][0]+box[0][2]]
        cv2.imwrite("foo2.jpg",crop)
        ans=tester.test(crop)
        ans=clf.predict(ans)
        e.sugar=n
        if (ans == 0):
            print "verified class 0"
        else:
            print "verification failed"
        return 1
    else:
        return 0
#print(pytesseract.image_to_string(Image.open('foo2.jpg')))


def green_f(A,n):
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
        cv2.rectangle(A,(box[0][0],box[0][1]),(box[0][0]+box[0][2],box[0][1]+box[0][3]),(255,255,255),2)
        cv2.putText(A,'Coffee powder',(box[0][0],box[0][1]), font, 1,(0,255,0),2)
        crop = A[box[0][1]:box[0][1]+box[0][3],box[0][0]:box[0][0]+box[0][2]]
        cv2.imwrite("foo1.jpg",crop)
        ans=tester.test(crop)
        ans=clf.predict(ans)
        e.milk=n

        if (ans == 1):
            print "verified class 1"
        else:
            print "verification failed"
        return 1
    else:
        return 0

#    print(pytesseract.image_to_string(Image.open('foo1.jpg')))

def red_f(A,n):
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
    if (max_area > 1500 ) :
        cnt=contours[ci]
        box[0][0],box[0][1],box[0][2],box[0][3] = cv2.boundingRect(cnt)
        cv2.rectangle(A,(box[0][0],box[0][1]),(box[0][0]+box[0][2],box[0][1]+box[0][3]),(255,255,255),2)
        cv2.putText(A,'Milk',(box[0][0],box[0][1]), font, 1,(0,0,255),2)
        
        #############
        crop = A[box[0][1]:box[0][1]+box[0][3],box[0][0]:box[0][0]+box[0][2]]
        cv2.imwrite("foo.jpg",crop)
        ans=tester.test(crop)
        ans=clf.predict(ans)
        e.coffee=n

        if (ans == 2):
            print "verified class 2"
    
        else:
            print "verification failed"
        return 1
    else:
        return 0

#        print(pytesseract.image_to_string(Image.open('foo.jpg')))


s1=0
s2=0
s3=0
nc=0
lu=0
ld=0
lr=0
ll=0
ob=0
cc=0
tes=0
yt=0
gt=0
rt=0
st=0
while(1):
    
    nah=0
    #e=pyttsx.init()
    box = [[0 for i in range(4)] for j in range(1)]
    # Take each frame
    _, frame = cap.read()
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    
    #cv2.imshow('frame',frame)
    #cv2.imshow('mask',mask)
   

    ret,thresh = cv2.threshold(mask,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('res',res)
    max_area=100
    ci=0
    for i in range(len(contours)):
        cnt=contours[i]
        area = cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci=i
    x=0
    y=0
    w=0
    h=0



    if(ci==0 or ci<0 or max_area<4000 ):
        cc=1
        ob=ob+1
        if(ob==15):
            system ('say Object not found try looking around')
            ob=0
        continue
    else:
        cc=0
        if (st==0):
            system("say Object found stay still")
            st=1
        print "Object found stay still"
        cnt=contours[ci]
        x,y,w,h = cv2.boundingRect(cnt)
        
        
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),6)        #boundry
        
        

        cv2.rectangle(frame,(x,y+int(h/3)),(x+w,y+int(2*h/3)),(0,0,0),2) #3
        
        cv2.rectangle(frame,(x,y),(x+w,y+int(h/3)),(0,0,0),2)     #1
        
        
        cv2.rectangle(frame,(x,y+int(2*h/3)),(x+w,y+h),(0,0,0),2) #5
        
        A = frame[y:y+int(h/3),x:x+w]
        B = frame[y+int(h/3):y+int(h*2/3),x:x+w]
        C = frame[y+int(2*h/3):y+h,x:x+w]
        
        yt=yellow_f(A,1)
        if yt!=1:
            yt=yellow_f(B,2)
        if yt!=1:
            yt=yellow_f(C,3)
        gt=green_f(A,1)
        if gt!=1:
            gt=green_f(B,2)
        if gt!=1:
            gt=green_f(C,3)
        rt=red_f(A,1)
        if rt!=1:
            rt=red_f(B,2)
        if rt!=1:
            rt=red_f(C,3)

        cv2.imshow("final",frame)




    if (max_area < 80000) :
        if cc==0:
            s1=s1+1
            if(s1==15):
                system('say Come closer')
                s1=0


    if(max_area>80000 and max_area< 170000 ):
        
        s2=s2+1
        if(s2==10 or tes==1):
            
            if tes==0:
                system('say perfect')
            tes=1
            if y<50 and (y+h)<380:
                nah=1
                lu=lu+1
                if(lu>1):
                    print "look up"
                    system('say look up')
                    lu=0
            elif y>100 and (y+h)>450:
                nah=1
                ld=ld+1
                if(ld>1):
                    print "look down"
                    system('say look down')
                    ld=0
            if x<100 and (x+w)<550:
                nah=1
                ll=ll+1
                if(ll>1):
                    print "look left"
                    system("say look left")
                    ll=0
            elif x>150 and (x+w)>580:
                nah=1
                lr=lr+1
                if(lr>1):
                    print "look right"
                    system("say look right")
                    lr=0

# print nah

            if nah==0 and yt==1 and gt==1 and rt==1:
            
                text = sp(e.coffee,e.milk,e.sugar)
                system("say  -v vicki "+ text)
                if (text == "coffee powder"):
                    system("say  -v vicki it is in rack "+ str(e.coffee))
                    hand(1)
                elif (text == "milk"):
                    system("say  -v vicki it is in rack "+ str(e.milk))
                    hand(2)
                elif (text == "sugar"):
                    system("say  -v vicki it is in rack "+ str(e.sugar))
                    hand(3)
                else:
                    system("say  -v vicki not in rack ")
                
                s2=0
    
    if(max_area > 170000):
        s3=s3+1
        if(s3==8):
            # system('say too close')
            s3=0
        print "Too close"

#   print max_area

#   cv2.drawContours(res,contours,ci,(0,255,0),-1)
#    cv2.imshow("frame2",res)




#       fun(A,B,C)
#        cv2.imshow("test",A)
#        cv2.imshow("test1",B)
#        cv2.imshow("test2",C)
#        cv2.imshow("test3",D)
#        cv2.imshow("test4",E)
#        cv2.imshow("test5",F)




#################################################################################################





    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()













