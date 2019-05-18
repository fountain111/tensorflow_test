#from __future__ import absolute_import,division,print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np

import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
print(max(test_labels))
print(train_images.shape)

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
#plt.show()

train_images = train_images/255

test_images = test_images/255

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.softmax)

])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']

)

model.fit(train_images,train_labels,epochs=5)

test_loss,test_acc = model.evaluate(test_images,test_labels)
print(test_loss,test_acc)

predictions = model.predict(test_images)
print(predictions.shape)
np.argmax(predictions[0])

img = test_images[0]

img = np.expand_dims(img,0)
img1 = np.expand_dims(img,1)

print(img.shape,img1.shape)