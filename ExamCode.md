#HaarCascades - Block #1

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import cv2
from scipy.spatial import distance


#location of haar casacade frontalface
face_model = cv2.CascadeClassifier(r':Insert path to haarcascade_frontalface_default.xml')

#sample image
img = cv2.imread(r':Insert sample image')

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





#Model Creation: Training, Testing and Validation - Block #2

import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Dense


#location of Training, Test and validation data
train_loc = r':Insert Path to Face Mask Dataset\Train'
test_loc = r':Insert Path to Face Mask Dataset\Test'
val_loc = r':Insert Path to Face Mask Dataset\Validation'


train_datagen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True, zoom_range=0.2,shear_range=0.2)
train_generator = train_datagen.flow_from_directory(directory=train_loc,target_size=(128,128),class_mode='categorical',batch_size=32)

val_datagen = ImageDataGenerator(rescale=1.0/255)
val_generator = train_datagen.flow_from_directory(directory=val_loc,target_size=(128,128),class_mode='categorical',batch_size=32)

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = train_datagen.flow_from_directory(directory=val_loc,target_size=(128,128),class_mode='categorical',batch_size=32)

#calling VGG-19, a deep learning image classifier
vgg19 = VGG19(weights='imagenet',include_top=False,input_shape=(128,128,3))

for layer in vgg19.layers:
    layer.trainable = False

#building Sequential model, adding mathematical function sigmoid so all values come back as either 1 or 0
model = Sequential()
model.add(vgg19)
model.add(Flatten())
model.add(Dense(2,activation='sigmoid'))
model.summary()


#calling optimizer "Adam", which is a stochastic optimization method
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics ="accuracy")


#cross validation process beginning using training and validation data
history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=len(train_generator)//32,
                              epochs=20,validation_data=val_generator,
                              validation_steps=len(val_generator)//32)


model.evaluate_generator(test_generator)


#Makes a prediction and uses it on a sample image 
sample_mask_img = cv2.imread(r':Insert sample image')
sample_mask_img = cv2.resize(sample_mask_img,(128,128))
plt.imshow(sample_mask_img)
sample_mask_img = np.reshape(sample_mask_img,[1,128,128,3])
sample_mask_img = sample_mask_img/255.0

model.predict(sample_mask_img)


#Label for the prediction 
mask_label = {0:'MASK FOUND',1:'NO MASK FOUND'}




#Prediction & Sample - Block #3

#Runs a sample image and test the prediction on it 
if len(faces)>=1:
    label = [0 for i in range(len(faces))]
    for i in range(len(faces)-1):
        for j in range(i+1, len(faces)):
            dist = distance.euclidean(faces[i][:2],faces[j][:2])
            if dist<MIN_DISTANCE:
                label[i] = 1
                label[j] = 1
    new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #colored output image
    for i in range(len(faces)):
        (x,y,w,h) = faces[i]
        crop = new_img[y:y+h,x:x+w]
        crop = cv2.resize(crop,(128,128))
        crop = np.reshape(crop,[1,128,128,3])/255.0
        mask_result = model.predict(crop)
        cv2.putText(new_img,mask_label[mask_result.argmax()],(x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,dist_label[label[i]],2)
        cv2.rectangle(new_img,(x,y),(x+w,y+h),dist_label[label[i]],1)
    plt.figure(figsize=(10,10))
    plt.imshow(new_img)
else:
    print("No face detected")
