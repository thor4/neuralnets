# Tensorflow basic classification: Classify images of clothing
# https://www.tensorflow.org/tutorials/keras/classification

import tensorflow as tf #tensorflow & tf.keras
#tf.keras is a high level library to build and train models in Tensorflow
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__) #2.4.0

#import fashion MNIST dataset containing 70k grayscale 28x28 clothing images in 10 categories
#similar to traditional digit MNIST dataset with 0-9 # categories. fashion is more challenging
#Both datasets are small and used to verify an algorithm works as expected. 
#Good starting points to test and debug code


