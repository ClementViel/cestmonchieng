import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import random
import cv2
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.layers import RandomFlip, RandomRotation
from tensorflow.keras.regularizers import L2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)

SEED = 42
IMG_HEIGHT = 299
IMG_WIDTH = 299
BATCH_SIZE = 32
EPOCHS = 50


def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def set_global_determinism(seed=SEED):
    set_seeds(seed)


# create dataset
train_ds = tf.keras.utils.image_dataset_from_directory(directory="/Users/clem/Projets/prog/cestmonchieng/dog_photo/train/",
                                                       label_mode='categorical',
                                                       color_mode='rgb',
                                                       batch_size=BATCH_SIZE,
                                                       image_size=(
                                                           IMG_HEIGHT, IMG_WIDTH),
                                                       shuffle=True,
                                                       seed=SEED,
                                                       validation_split=0.4,
                                                       subset='training')
validation_ds = tf.keras.utils.image_dataset_from_directory(directory="/Users/clem/Projets/prog/cestmonchieng/dog_photo/train/",
                                                            label_mode='categorical',
                                                            color_mode='rgb',
                                                            batch_size=BATCH_SIZE,
                                                            image_size=(
                                                                IMG_HEIGHT, IMG_WIDTH),
                                                            shuffle=True,
                                                            seed=SEED,
                                                            validation_split=0.4,
                                                            subset='validation')

val_batches = tf.data.experimental.cardinality(validation_ds)
test_ds = validation_ds.take((2*val_batches) // 3)
val_ds = validation_ds.skip((2*val_batches) // 3)
num_classes = 2
class_names = train_ds.class_names
print(class_names)


# Create the Convolutional Neural Network. Classsical NN for object detection
model = Sequential([
    Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((3, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((3, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((3, 3)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.categorical_accuracy]
)

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)

df = pd.DataFrame(history.history)
acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
#plt.show()
results = model.evaluate(test_ds, batch_size=32, verbose=1)
print("test loss, test acc:", results)
model.save('/Users/clem/Projets/prog/cestmonchieng/monchieng.keras')
tf.saved_model.save(
    model, "saved_model")
# Export the keras model to a saved model format
# model.export("saved_model")

# Convert the saved model to TensorFlow Lite
# converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
