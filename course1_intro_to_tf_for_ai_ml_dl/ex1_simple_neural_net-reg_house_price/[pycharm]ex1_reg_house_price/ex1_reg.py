import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("ex1_reg.h5")
print(model.predict([7.0]))
