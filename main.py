import os
import cv2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
from tensorflow.keras.regularizers import l2
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator


down = []
normals = []

dataset_path = '/home/down' # PATH OF THE DATASET

for path, subdirs, files in os.walk(dataset_path):
    if path.endswith("healthy"):
        images = os.listdir(path)
        for img in enumerate(images):
            image = cv2.imread(path + "/" + img[1], 1)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                normals.append(image)
    if path.endswith("downSyndrome"):
        images = os.listdir(path)
        for img in enumerate(images):
            image = cv2.imread(path + "/" + img[1], 1)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                down.append(image)

dataset = []
minmaxscaler = MinMaxScaler()

for image in down:
    image = minmaxscaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
    dataset.append([image, 0])
    
for image in normals:
    image = minmaxscaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
    dataset.append([image, 1])

random.shuffle(dataset)

data = []
labels = []
for image in dataset:
    data.append(image[0])
    labels.append(image[1])

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, train_size=0.6, random_state=42)
    
train = np.array(X_train)
vali = np.array(X_test)
y_tra = np.array(y_train)
y_val = np.array(y_test)

import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout,Input
from keras.models import Sequential

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.002), input_shape=(300, 300, 3)))
model.add(MaxPooling2D())
model.add(Dropout(0.5))

model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

model.summary()



logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(train, y_tra, epochs=20, validation_data=(vali, y_val), callbacks=[tensorboard_callback], batch_size=64)

plt.plot(hist.history['accuracy'], label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(hist.history['loss'], label='Training Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save('model.h5')








