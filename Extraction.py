import cv2
import numpy as np
import csv
import os
#from scipy import mode

#os.remove('data.csv')
#os.remove('target.csv')

Humom1=0
means=0
hist=0
stats=0
Humom2=0
cn=0
class Extract(object):

    def ext(self,img,classnum):
        cn=classnum
		#Hu-moments Extraction
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        blur = cv2.bilateralFilter(thresh,15,80,80)
        Humom1=cv2.HuMoments(cv2.moments(blur)).flatten()
		#RGB Means extraction
        means = cv2.mean(img)
        means = means[:3]
		#Histogram extraction
        #hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        #hist = hist.flatten()
        hist,bins = np.histogram(img.flatten(),256,[0,256])
        cdf = hist.cumsum()
		#Contour Hu
        gray2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        thresh=cv2.adaptiveThreshold(gray2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        blur = cv2.bilateralFilter(thresh,15,80,80)
        gray_lap = cv2.Laplacian(blur,cv2.CV_16S,ksize = 3,scale = 1,delta = 0)
        dst = cv2.convertScaleAbs(gray_lap)
        Humom2=cv2.HuMoments(cv2.moments(dst)).flatten()
		#Stats Extraction
        (means2, stds) = cv2.meanStdDev(img)
        stats = np.concatenate([means2, stds]).flatten()
		#Class appending
        total1=np.append(Humom1,means)
        total2=np.append(total1,cdf)
        total3=np.append(total2,stats)
        total=np.append(total3,Humom2)
        cn=np.append(cn,0)
		#most_intensity=np.append(most_intensity,classnum)
		
        with open('data.csv', 'ab')as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(total)
        with open('target.csv', 'ab')as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(cn)
    def test(self,img):
        #Hu-moments Extraction
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        blur = cv2.bilateralFilter(thresh,15,80,80)
        Humom1=cv2.HuMoments(cv2.moments(blur)).flatten()
        #RGB Means extraction
        means = cv2.mean(img)
        means = means[:3]
        #Histogram extraction
        #hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        #hist = hist.flatten()
        hist,bins = np.histogram(img.flatten(),256,[0,256])
        cdf = hist.cumsum()
        #Contour Hu
        gray2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        thresh=cv2.adaptiveThreshold(gray2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        blur = cv2.bilateralFilter(thresh,15,80,80)
        gray_lap = cv2.Laplacian(blur,cv2.CV_16S,ksize = 3,scale = 1,delta = 0)
        dst = cv2.convertScaleAbs(gray_lap)
        Humom2=cv2.HuMoments(cv2.moments(dst)).flatten()
        #Stats Extraction
        (means2, stds) = cv2.meanStdDev(img)
        stats = np.concatenate([means2, stds]).flatten()
        #Class appending
        total1=np.append(Humom1,means)
        total2=np.append(total1,cdf)
        total3=np.append(total2,stats)
        total=np.append(total3,Humom2)
        total1=[float(i) for i in total]
        X=np.array(total1).reshape((1,-1))
        return X


