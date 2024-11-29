import tensorflow as tf
from keras.models import model_from_json

model = tf.keras.models.load_model(
    'monchieng.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
