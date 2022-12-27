import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import tensorflow_hub as hub

image_size = (224, 224) #(400, 600)
batch_size = 32

def predict(model,imgPath):
    img = keras.preprocessing.image.load_img(imgPath, target_size=image_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array /= 255
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    #'drawings', 'hentai', 'neutral', 'porn', 'sexy'
    return predictions

if __name__ == '__main__':
    reconstructed_model = keras.models.load_model("mobilenet_v2_140_224")
    #reconstructed_model = keras.models.load_model("mobilenet_v2_140_224", custom_objects={'KerasLayer': hub.KerasLayer},compile=False)
    result = predict(reconstructed_model, "../../input/nsfw/data/porn/FF207B93-2980-4D03-9649-CDBADC806CFE.jpg")
    print(result)