import tensorflow as tf
from tensorflow import keras
import numpy as np

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = keras.models.load_model("models/ex3_recode.h5")

x_test = (x_test.reshape(10000, 28, 28, 1 ))
x_test = x_test/255.0

preds = model.predict(x_test[:3])
print(preds)
print(np.argmax(i) for i in preds)
print(y_test[:3])
