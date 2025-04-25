import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import matplotlib.pyplot as plt
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.layers import RandomFlip, RandomRotation
from tensorflow.keras.regularizers import L2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
IMG_HEIGHT = 299
IMG_WIDTH = 299
BATCH_SIZE = 32


def isitmonchieng(num):
    if num > 0.9:
        print("c'est mon chieng")
    else:
        print("pas mon chieng")


# model = keras.models.load_model(
#    "/Users/clem/Projets/prog/cestmonchieng/monchieng.keras")

interpreter = tf.lite.Interpreter(
    model_path="/Users/clem/Projets/prog/cestmonchieng/model_meta/model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
img = keras.utils.load_img(
    "/Users/clem/Downloads/PXL_20250117_145923956.jpg", target_size=(IMG_HEIGHT, IMG_WIDTH))

# img = keras.utils.load_img(
#    "/Users/clem/Downloads/not_java.jpg", target_size=(IMG_HEIGHT, IMG_WIDTH))

img_array = keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, 0)  # Create batch axis
interpreter.set_tensor(input_details[0]['index'], img_array)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print("result ", output_data)


# predictions = model.predict(img_array)
# print(predictions[0][0])
# print(predictions[0][1])

# isitmonchieng(predictions[0][0])
