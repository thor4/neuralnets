# Loading images tutorial
# https://www.tensorflow.org/tutorials/load_data/images

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds

print(tf.__version__)

import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url, 
                                   fname='flower_photos', 
                                   untar=True) #downloads photos and returns path to them
data_dir = pathlib.Path(data_dir) #converts str path to os-appropriate format (windows/unix)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count) #3670 flower photos

roses = list(data_dir.glob('roses/*')) #stores list of all paths to flowers of class 'roses'
PIL.Image.open(str(roses[0])) #convert 1st PosixPath in list to str then open with PIL.Image class


#my own implementation:
curr_dir = os.getcwd() #make sure I'm in CNN project folder
curr_dir = pathlib.Path(curr_dir) #converts str path to os-appropriate format (windows/unix)
len(list(curr_dir.glob('**/*.png'))) #locates all the gabor images, ** means this dir and & subdir's
#100 gabors of each class + base & gabor in main dir = 202
clock = list(curr_dir.glob('images/clock/*')) #stores list of all paths to gabors of class 'clockwise'
PIL.Image.open(str(clock[0])) #convert 1st PosixPath in list to str then open with PIL.Image class