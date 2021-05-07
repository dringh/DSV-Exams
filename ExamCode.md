import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import cv2
from scipy.spatial import distance


#location of haar casacade frontalface
face_model = cv2.CascadeClassifier(r'D:/Users/Martin/PycharmProjects/pythonProject1/haarcascade_frontalface_default.xml')

#sample image
img = cv2.imread(r'C:\Users\Martin\Desktop\mette.png')

#image grayscale 
img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

#This line helps reducing the amount of false positives
faces = face_model.detectMultiScale(img,scaleFactor=1.05, minNeighbors=6) #returns a list of (x,y,w,h) tuples


out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #colored output image


#plotting coordinates of the square for haar cascade
for (x,y,w,h) in faces:
    cv2.rectangle(out_img,(x,y),(x+w,y+h),(255,255,0),1)
plt.figure(figsize=(12,14))
plt.imshow(out_img)
