import os
import tensorflow as tf
import cv2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random

minmaxscaler = MinMaxScaler()
downSyndrome = []
non_downSyndrome = []

dataset_path = '/down'

for path, subdirs, files in os.walk(dataset_path):
    if path.endswith("healthy"):
        images = os.listdir(path)
        for img in enumerate(images):
            image = cv2.imread(path + "/" + img[1], 1)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                non_downSyndrome.append(image)
    if path.endswith("downSyndrome"):
        images = os.listdir(path)
        for img in enumerate(images):
            image = cv2.imread(path + "/" + img[1], 1)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                downSyndrome.append(image)

model_path = '/model.h5'
model = tf.keras.models.load_model(model_path, compile=False)
model.summary()

# TESTING THE MODEL
r = random.randint(0, 500)
image = np.array(downSyndrome[r])
pic = minmaxscaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
plt.imshow(pic)
pic = tf.expand_dims(pic, axis=0)
prediction = model.predict(pic)
print(prediction)
h = prediction
if h < (1 - h):
    print("Has down syndrome")        
else:
    print("Does not have down syndrome")
