import cv2
import numpy as np
from numpy import sqrt,arccos,rad2deg
from os import system
from extract2 import Extract
import csv
from sklearn.naive_bayes import GaussianNB
import sklearn
import time
class hh:
    n=0
    m=0
    an=0
tester=Extract()
csv = np.genfromtxt ('target2.csv', delimiter=",")
target = csv[:,0]

y=[int(i)for i in target]
Y=np.array(y)
import csv
with open('data2.csv', 'rb') as f:
    reader = csv.reader(f)
    lst = list(reader)
    x = [[(float(j)) for j in i] for i in lst]
    x = [np.array(i) for i in x]
X=np.array(x)
clf = GaussianNB()
clf.fit(X,Y)
###################################


def veri(lm):
    h=hh()
    cap1 = cv2.VideoCapture(1)
    
    while( cap1.isOpened() ) :

        ret,img = cap1.read()
        box = [[0 for i in range(4)] for j in range(1)]
        box[0][0]=160
        box[0][1]=140
        box[0][2]=300
        box[0][3]=240
        cv2.rectangle(img,(box[0][0],box[0][1]),(box[0][0]+box[0][2],box[0][1]+box[0][3]),(255,255,255),2)
        crop = img[box[0][1]:box[0][1]+box[0][3],box[0][0]:box[0][0]+box[0][2]]
        cv2.imshow("hii",img)
        ans=tester.test(crop)
        h.an=clf.predict(ans)
        print h.an
        print lm
        if h.an==0:
            h.n=h.n+1
            if h.n==10 and lm==2:
                system(" say -v vicki Nice job, you have picked up coffee powder.")
                system("say -v vicki Happy coffee, bye bye")
                time.sleep(120)

            if h.m!=0:
                h.m=0
        if h.an==1:
            h.m=h.m+1
            if h.m==10 and lm==3:
                system("say -v vicki Nice job, you have picked up sugar.")
                system("say -v vicki Happy coffee, bye bye")
                time.sleep(120)

            if h.n!=0:
                h.n=0
