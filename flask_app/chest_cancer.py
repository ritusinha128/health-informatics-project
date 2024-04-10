from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

def wb(channel, perc = 0.05):
    mi, ma = (np.percentile(channel, perc), np.percentile(channel,100.0-perc))
    channel = np.uint8(np.clip((channel-mi) * 255.0 / (ma-mi), 0, 255))
    return channel

def read_image(image_path):
    image = cv2.imread(image_path)
    # performing white balance
    imWB  = np.dstack([wb(channel, 0.05) for channel in cv2.split(image)] )
    gray_image = cv2.cvtColor(imWB, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(img, (224, 224))
    return image

def predict_image(image_path): 
    image = read_image(image_path)
    image = image[np.newaxis, ...]
    model = load_model('model/ChestCancer.h5')
    prediction = np.argmax(model(image))
    target_names = ['COVID-19', 'Normal', 'Pneumonia-Bacterial Infection']
    return target_names[prediction]

