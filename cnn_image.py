# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 18:55:54 2020

@author: chris
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# Import TensorFlow modules
import tensorflow as tf
from tensorflow import keras

# Import numerical modules
import numpy as np
import matplotlib.pyplot as plt

print('TensorFlow Version: {}'.format(tf.__version__))

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Import fashion data & normalise
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# Data key
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Display 25 test images
plt.figure(figsize=(12,12))
for im in range (25):
    plt.subplot(5,5,im+1)
    plt.imshow(test_images[im], cmap='Greys', interpolation='nearest')
    plt.xticks([])
    plt.xlabel(class_names[test_labels[im]])
    plt.yticks([])
plt.show()

# Create CNN parameters
model = keras.Sequential([keras.layers.Flatten(input_shape=[28,28]),
                          keras.layers.Dense(128,activation='relu'),
                          keras.layers.Dense(128,activation='relu'),
                          keras.layers.Dense(10,activation='softmax')])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(train_images, train_labels, epochs=5)
"""
# Test model
test_im = 25
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Test accuracy = {}'.format(test_acc))
predictions = model.predict(test_images)
im_prediction = class_names[np.argmax(predictions[test_im])]
im_actual = class_names[test_labels[test_im]]
print('Actual: {}'.format(im_actual))
print('Prediction: {}'.format(im_prediction))

# Show confidence of image predictions
for i in range(25):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions[i],  test_labels)
    plt.show()
    
# Make prediction about single image
for i in range(50):
    img = test_images[i]
    img = np.expand_dims(img, 0)
    predictions_single = model.predict(img)
    print(class_names[np.argmax(predictions_single)])
"""

