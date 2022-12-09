#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
import os
import numpy as np

def load_model_file(model_path, model_weights):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    model_file = open(model_path, 'r')#open using os
    loaded_model_json = model_file.read() #load json file in variable
    model_file.close() #close the opener
    loaded_model = model_from_json(loaded_model_json) #load model
    print(f"Model Loaded successfully from {model_path}")
    loaded_model.load_weights(model_weights)
    print(f"Model weights loaded successfully from {model_weights}")
    loaded_model.compile(optimizer= 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    metrics=['accuracy'])
    print("Model Compiled Successfully")
    return loaded_model

def prediction (model,img):
    #img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    #img_array = tf.keras.preprocessing.image.img_to_array(img.np())
    class_names = ["Healthy", "Resistant", "Susceptible"]
    img_array = tf.expand_dims(img, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round (100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

def image_loader (path):
    from PIL import Image
    im = Image.open(path)
    im = im.resize((320, 320), Image.Resampling.LANCZOS)
    #im.show()
    im = np.asarray(im)
    print(f"Image {path} loaded successfully")
    return im


# img_path = 'r.jpg'
# image = image_loader (img_path)
#
#
# # In[74]:
#
#
# model_path = 'model.json'
# model_weights = ("best_model.hdf5")
# loaded_model = load_model_file (model_path, model_weights)
#
#
# # In[75]:
#
#
# predic_class, confidence = prediction(loaded_model, image)
# print(f"\nPredicted Class =\t {predic_class} ")
# print(f"Confidence =\t\t {confidence} ")


# In[ ]:
