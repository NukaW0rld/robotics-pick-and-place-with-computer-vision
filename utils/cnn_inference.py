import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2

class_names = ['Hammer', 'Pliers', 'Screwdriver', 'Wrench']

stored_model_path = 'saved_models/tool_classifier_04_20_20_01.keras'

loadmodel = keras.models.load_model(stored_model_path)

def cnn_inference(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320,240))
    image_tensor = tf.expand_dims(image, axis=0)
    result = loadmodel(image_tensor)
    return class_names[np.argmax(result)]