import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import pandas
import h5py
import glob
from keras.initializers import glorot_uniform

h5file =  "model.h5"

with h5py.File(h5file,'r') as fid:
     model = load_model(fid)

def get_filenames():
    global path
    path = r"test"
    return os.listdir(path)

def autoroi(img):

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray_img, 130, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=5)

    contours, hierarchy = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    roi = img[y:y+h, x:x+w]

    return roi


def prediction():
    img = cv2.imread("Img02.png")
    img = autoroi(img)
    img = cv2.resize(img, (96, 96))
    img = np.reshape(img, [1, 96, 96, 3])
    img = tf.cast(img, tf.float64)

    prediction = model.predict(img)
    print(prediction)
    Class = prediction.argmax(axis=1)
    print(Class)


    return(Class)


finalPrediction = prediction()
print(finalPrediction[0])
print("Final Prediction = ", finalPrediction)
if (finalPrediction == 0):
    print("No TB.")
else:
    print("TB.")