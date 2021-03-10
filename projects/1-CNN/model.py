## CNN model to dissociate confidence from accuracy
#use transfer learning tutorial here: https://www.tensorflow.org/tutorials/images/transfer_learning

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os

tf.__version__ #2.4.0

#load images
path = os.getcwd() #make sure I'm in CNN project folder

#init vars
BATCH_SIZE = 32
IMG_SIZE = (160, 160)
 
 # instantiate a MobileNet V2 model pre-loaded with weights trained on ImageNet.
 # The include_top=False means you load a network that doesn't include the classification layers at the 
 # top, which is ideal for feature extraction. We want the "bottleneck layer", which is the very last 
 # just before the flatten operation. It has more general features.
# Create the base model from MobileNet V2 model pre-trained on Imagenet dataset (1.4M images / 1000 classes)
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

#NEED TO GENERATE TEST IMAGES
#THEN RUN THROUGH THE TUTORIAL ON MY OWN. During tutorial, I can update this script accordingly based on what I learn.