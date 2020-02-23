# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:18:46 2020

@author: chris
"""

# Import TensorFlow libraries
import tensorflow as tf
from tensorflow import keras

# Import mathematical and graphing libraries
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data set
mnist_data = keras.datasets.mnist
(x_images, x_labels), (y_images, y_labels) = mnist_data.load_data()

# Image checker function
def show_image(im_arr, im_start):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,1+i)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(im_arr[im_start + i], cmap=plt.cm.binary)
        
show_image(x_images, 500)