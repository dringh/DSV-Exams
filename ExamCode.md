# DSV-Exams
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from scipy.spatial import distance
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Dense

#import os
#for dirname, _, filenames in os.walk(r'D:\Users\Martin\PycharmProjects\pythonProject1\Kaggle'):
# for filename in filenames:
# print(os.path.join(dirname, filename))

#loads the XML file
face_model = cv2.CascadeClassifier("/Users/dinahansen/PycharmProjects/DataScience/DSVexam/haarcascade_frontalface_default.xml")

import matplotlib.pyplot as plt
#trying it out on a sample image
img = cv2.imread("/Users/dinahansen/PycharmProjects/DataScience/DSVexam/people.png")

img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

faces = face_model.detectMultiScale(img,scaleFactor=1.05, minNeighbors=6) #returns a list of (x,y,w,h) tuples

out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #colored output image

#plotting
for (x,y,w,h) in faces:
    cv2.rectangle(out_img,(x,y),(x+w,y+h),(0,0,255),5)
plt.figure(figsize=(12,12))
plt.imshow(out_img)


train_dir = "/Users/dinahansen/PycharmProjects/DataScience/DSVexam/Face Mask Dataset/Train"
test_dir = "/Users/dinahansen/PycharmProjects/DataScience/DSVexam/Face Mask Dataset/Test"
val_dir = "/Users/dinahansen/PycharmProjects/DataScience/DSVexam/Face Mask Dataset/Validation"

train_datagen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True, zoom_range=0.2,shear_range=0.2)
train_generator = train_datagen.flow_from_directory(directory=train_dir,target_size=(128,128),class_mode='categorical',batch_size=32)

val_datagen = ImageDataGenerator(rescale=1.0/255)
val_generator = train_datagen.flow_from_directory(directory=val_dir,target_size=(128,128),class_mode='categorical',batch_size=32)

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = train_datagen.flow_from_directory(directory=val_dir,target_size=(128,128),class_mode='categorical',batch_size=32)

vgg19 = VGG19(weights='imagenet',include_top=False,input_shape=(128,128,3))

for layer in vgg19.layers:
    layer.trainable = False

model = Sequential()
model.add(vgg19)
model.add(Flatten())
model.add(Dense(2,activation='sigmoid'))
model.summary()

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics ="accuracy")

history = model.fit_generator(generator=train_generator,
steps_per_epoch=len(train_generator)//32,
epochs=20,validation_data=val_generator,
validation_steps=len(val_generator)//32)
