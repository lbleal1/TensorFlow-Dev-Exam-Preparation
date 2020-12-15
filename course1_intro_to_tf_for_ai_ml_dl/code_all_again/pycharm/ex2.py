import tensorflow as tf
from tensorflow import keras
import numpy as np

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = keras.models.load_model("models/ex2_recode.h5")
#model.summary()
x_test = x_test/255.0
print(model.evaluate(x_test, y_test))
print(model.predict(x_test[:1]))
print(np.argmax(model.predict(x_test[:1])))
print(y_test[0])