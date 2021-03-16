## CNN model to dissociate confidence from accuracy
#use transfer learning tutorial here: https://www.tensorflow.org/tutorials/images/transfer_learning

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os

tf.__version__ #2.4.0

#my own images for customizing the pretrained model:
curr_dir = os.getcwd() #make sure I'm in CNN project folder
curr_dir = pathlib.Path(curr_dir) #converts str path to os-appropriate format (windows/unix)
len(list(curr_dir.glob('**/*.png'))) #locates all the gabor images, ** means this dir and & subdir's
#100 gabors of each class + base & gabor in main dir = 202
clock = list(curr_dir.glob('images/clock/*')) #stores list of all paths to gabors of class 'clockwise'
PIL.Image.open(str(clock[0])) #convert 1st PosixPath in list to str then open with PIL.Image class

#only needed for tutorial purposes to get images for customizing the pretrained model:
from tensorflow.keras.preprocessing import image_dataset_from_directory
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
# '/home/jovyan/.keras/datasets/cats_and_dogs.zip'
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
# '/home/jovyan/.keras/datasets/cats_and_dogs_filtered'
train_dir = os.path.join(PATH, 'train')
# '/home/jovyan/.keras/datasets/cats_and_dogs_filtered/train'
# there is a 'cats' and 'dogs' folder in here with 1000 images a piece (2/3)
validation_dir = os.path.join(PATH, 'validation')
# '/home/jovyan/.keras/datasets/cats_and_dogs_filtered/validation'
# there is a 'cats' and 'dogs' folder in here with 500 images a piece (1/3)

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

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