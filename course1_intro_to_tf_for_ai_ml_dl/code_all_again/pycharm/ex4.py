import os
import zipfile
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np

model = keras.models.load_model("models/ex4_recode.h5")

path = 'tmp2/happy-or-sad/happy/happy1-00.png'
img = image.load_img(path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes[0])
if classes[0]>0.5:
    print("sad")
else:
    print("happy")