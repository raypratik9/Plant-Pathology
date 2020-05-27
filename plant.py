from tensorflow.keras.layers import Dense, Flatten, Conv2D ,MaxPool2D
from tensorflow.keras import Model
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import random
import math
import os
import cv2
import matplotlib.pyplot as plt
IMAGE_DIR = 'dataset\images'
BATCH_SIZE = 8
EPOCHS = 20
SAMPLE_LEN = 100
IMG_SIZE=1000

trainDf = pd.read_csv('dataset/train.csv')
testDf = pd.read_csv('dataset/test.csv')
trainDf

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img=mpimg.imread('dataset/images/Train_0.jpg')
img = cv2.resize(img, (200,200))
imgplot = plt.imshow(img)
plt.show()

Batches = list(trainDf['image_id'])
x,y=[],[]
for i in Batches:
    imagePath = "dataset/images/" + i + ".jpg"
    img = mpimg.imread(imagePath)
    img = cv2.resize(img, (200,200))
    img=img/255
    labelsDf = trainDf[trainDf['image_id']==i]
    labels = [int(labelsDf['healthy']),int(labelsDf['multiple_diseases']),int(labelsDf['rust']),int(labelsDf['scab'])]
    x.append(img)
    print(i)
    y.append(labels)

x=np.array(x)
y=np.array(y)

model=Sequential()

model.add(Conv2D(32,(3,3),input_shape=x.shape[1:]))
model.add(Activation("relu"))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),input_shape=x.shape[1:]))
model.add(Activation("relu"))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),input_shape=x.shape[1:]))
model.add(Activation("relu"))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),input_shape=x.shape[1:]))
model.add(Activation("relu"))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

history=model.fit(x,y,batch_size=100,epochs=15,verbose=1
                  ,validation_split=0.3)


model.save('plant_pathology.h5')


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

