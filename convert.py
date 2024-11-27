import tensorflow as tf
from keras.models import load_model
import numpy as np

# Load the Keras model
model = load_model("model.keras")

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('emotion_model.tflite', 'wb') as f:
    f.write(tflite_model)