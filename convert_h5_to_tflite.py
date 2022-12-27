import tensorflow as tf

model = tf.keras.models.load_model('/content/animall_person_other_v2_fine_tuned.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("animall_person_other_v2_fine_tuned.tflite", "wb").write(tflite_model)