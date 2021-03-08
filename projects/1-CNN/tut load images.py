# Loading images tutorial
# https://www.tensorflow.org/tutorials/load_data/images

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds

print(tf.__version__)

path = os.getcwd() #make sure I'm in CNN project folder