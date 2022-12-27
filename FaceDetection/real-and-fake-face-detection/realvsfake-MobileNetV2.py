from keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense,BatchNormalization, Flatten, MaxPool2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
from tensorflow import keras

dataset_path = "../../input/real_and_fake_face/"

def training_data():
    data_with_aug = ImageDataGenerator(horizontal_flip=True,
                                    vertical_flip=False,
                                    rescale=1./255,
                                    validation_split=0.2)

    train = data_with_aug.flow_from_directory(dataset_path,
                                            class_mode="binary",
                                            target_size=(96, 96),
                                            batch_size=32,
                                            subset="training")

    val = data_with_aug.flow_from_directory(dataset_path,
                                            class_mode="binary",
                                            target_size=(96, 96),
                                            batch_size=32,
                                            subset="validation"
                                            )

    # MobileNetV2
    mnet = MobileNetV2(include_top = False, weights = "imagenet" ,input_shape=(96,96,3))
    #https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5

    tf.keras.backend.clear_session()

    model = Sequential([mnet,
                        GlobalAveragePooling2D(),
                        Dense(512, activation = "relu"),
                        BatchNormalization(),
                        Dropout(0.3),
                        Dense(128, activation = "relu"),
                        Dropout(0.1),
                        # Dense(32, activation = "relu"),
                        # Dropout(0.3),
                        Dense(2, activation = "softmax")])

    model.layers[0].trainable = False

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics="accuracy")

    def scheduler(epoch):
        if epoch <= 2:
            return 0.001
        elif epoch > 2 and epoch <= 15:
            return 0.0001 
        else:
            return 0.00001

    lr_callbacks = tf.keras.callbacks.LearningRateScheduler(scheduler)

    model.fit_generator(train,
                        epochs=20,
                        callbacks=[lr_callbacks],
                        validation_data=val)
    return model

def predict(model, imgPath):
    img = keras.preprocessing.image.load_img(imgPath, target_size=(96, 96))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = predictions[0][0]
    return 100 * (1 - score), 100 * score

def predict_folder(model):
    count = 0
    folder_path = os.path.join(dataset_path, "training_real")
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        ext = os.path.splitext(fpath)[-1].lower()
        if ext == ".jpg":
            real,fake = predict(model, fpath)
            if real > fake:
                count=count+1
            else:
                print(fname)
            print(
                "This image is %.2f percent fake and %.2f percent real."
                % (fake, real)
            )

    print(count/len(os.listdir(folder_path)))

#model=training_data()
#model.save("MobileNetV2/")

model = keras.models.load_model("MobileNetV2/")
real,fake = predict(model,"../../input/real_and_fake_face/training_fake/easy_85_1111.jpg")

print(
                "This image is %.2f percent fake and %.2f percent real."
                % (fake, real)
            )