import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from pathlib import Path
import sys

image_size = (224, 224)
batch_size = 32
epochs = 25
dataset_dir = Path("../../input/identity/")
model_name = "models_29_08_2022"

def predict(model,imgPath):
    img = keras.preprocessing.image.load_img(imgPath, target_size=image_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = predictions[0]
    return score * 100

if __name__ == '__main__':
    reconstructed_model = keras.models.load_model(model_name)
    result = predict(reconstructed_model, "../../input/face/4.jpg")
    print(result)
    # result = predict(reconstructed_model, "../../input/identity/training_false/easy_202_0011.jpg")
    # print(result)

    