import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from pathlib import Path

image_size = (180, 180)
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE
epochs = 50
dataset_dir = Path("PetImages")
model_name = "CatOrDog_CPU"

def prepare(ds, shuffle=False, augment=False):
    resize_and_rescale = tf.keras.Sequential([layers.Resizing(image_size[0],image_size[1]),layers.Rescaling(1.0 / 255)])

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ])

    # Resize and rescale all datasets.
    ds = ds.map(lambda x, y: (resize_and_rescale(x), y))

    if shuffle:
        ds = ds.shuffle(1000)

    # Batch all datasets.
    ds = ds.batch(batch_size)

    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y))

    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)

def make_model(input_shape, num_classes):
    model = tf.keras.Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
    ])
    return model

def train_data():
    # Datasets
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(dataset_dir),
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(dataset_dir),
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )

    train_ds = prepare(train_ds, shuffle=True, augment=True)
    val_ds = prepare(val_ds)

    # Create Model
    model = make_model(input_shape=image_size + (3,), num_classes=2)
    keras.utils.plot_model(model, show_shapes=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
    ]
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.fit(
        train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
    )

    return model

    def predict(model,imgPath):
        img = keras.preprocessing.image.load_img(imgPath, target_size=image_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        predictions = model.predict(img_array)
        score = predictions[0]
        print(
            "This image is %.2f percent cat and %.2f percent dog."
            % (100 * (1 - score), 100 * score)
        )

# def visualize_data(train_ds, val_ds):
#     for images, labels in train_ds.take(1):
#         for i in range(9):
#             print(labels[i])
#             cv2.imwrite(str(i)+".jpg", images[i].numpy().astype("uint8"))
#             #plt.imshow(images[i].numpy().astype("uint8"))

if __name__ == '__main__':
    #filter_corrupted_images()

    model = train_data()
    model.save(model_name)

    #predict(model, "PetImages/Cat/6779.jpg")