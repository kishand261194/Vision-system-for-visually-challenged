import cv2
import numpy as np
from os import system
from PIL import Image
from extract2 import Extract


train = Extract()
cap = cv2.VideoCapture(1)
font = cv2.FONT_HERSHEY_SIMPLEX

#30-40 yellow
#40-50 green



def red_f(A):
    
    
    #   train.ext(A,1)
        box = [[0 for i in range(4)] for j in range(1)]
        box[0][0]=160
        box[0][1]=140
        box[0][2]=300
        box[0][3]=240
        cv2.rectangle(A,(box[0][0],box[0][1]),(box[0][0]+box[0][2],box[0][1]+box[0][3]),(255,255,255),2)
        crop = A[box[0][1]:box[0][1]+box[0][3],box[0][0]:box[0][0]+box[0][2]]
        train.ext(crop,1 )



while(1):
    


    _, frame = cap.read()
    box = [[0 for i in range(4)] for j in range(1)]
    box[0][0]=160
    box[0][1]=140
    box[0][2]=300
    box[0][3]=240
    cv2.rectangle(frame,(box[0][0],box[0][1]),(box[0][0]+box[0][2],box[0][1]+box[0][3]),(255,255,255),2)
    cv2.imshow("final",frame)
    if cv2.waitKey(50) == ord('n'):
        break
    
while(1):
    
    
    
    _, frame = cap.read()
    box = [[0 for i in range(4)] for j in range(1)]
    box[0][0]=160
    box[0][1]=140
    box[0][2]=300
    box[0][3]=240
    cv2.rectangle(frame,(box[0][0],box[0][1]),(box[0][0]+box[0][2],box[0][1]+box[0][3]),(255,255,255),2)
    red_f(frame)
    cv2.imshow("final",frame)
    

    
    
    
    
    
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()













