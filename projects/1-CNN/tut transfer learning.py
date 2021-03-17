# Transfer learning and fine-tuning tutorial
# https://www.tensorflow.org/tutorials/images/transfer_learning

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os

tf.__version__ #2.4.0

#my own images for customizing the pretrained model:
#import pathlib
curr_dir = os.getcwd() #make sure I'm in CNN project folder
# '/workspaces/neuralnets/projects/1-CNN' (equiv to PATH var below)
train_dir = os.path.join(curr_dir, 'images/train')
# '/workspaces/neuralnets/projects/1-CNN/images/train'
# there is a 'clock' and 'cclock' folder in here with 60 images a piece (3/5)
validation_dir = os.path.join(curr_dir, 'images/validation')
# '/workspaces/neuralnets/projects/1-CNN/images/validation'
# there is a 'clock' and 'cclock' folder in here with 20 images a piece (1/5)
test_dir = os.path.join(curr_dir, 'images/test')
# '/workspaces/neuralnets/projects/1-CNN/images/validation'
# there is a 'clock' and 'cclock' folder in here with 20 images a piece (1/5)
BATCH_SIZE = 60 #stick with one iteration for training and validation per optimal batch guidance here:
# https://ai.stackexchange.com/questions/8560/how-do-i-choose-the-optimal-batch-size
# practically, this means only one update of gradient and neural network parameters
IMG_SIZE = (170, 170) #should all already be this size so probably redundant but good to be sure
train_dataset = image_dataset_from_directory(train_dir,
                                             color_mode="grayscale", #rgb by default, save 1 chan instead of 3
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE) #Found 134 files belonging to 2 classes.
validation_dataset = image_dataset_from_directory(validation_dir,
                                                  color_mode="grayscale", #rgb by default, save 1 chan instead of 3
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE) #Found 66 files belonging to 2 classes.
test_dataset = image_dataset_from_directory(validation_dir,
                                                  color_mode="grayscale", #rgb by default, save 1 chan instead of 3
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE) #Found 66 files belonging to 2 classes.
#show first nine images and labels from training set:
class_names = train_dataset.class_names #extract class names previous function inferred from subdir's
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1): #load first iteration batch from training dataset
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1) #setup axis on a 3x3 grid
    plt.imshow(images[i].numpy().astype("uint8"),cmap='gray') #plot each image
    plt.title(class_names[labels[i]]) #output associated label for chosen image
    plt.axis("off")
#if I need to use more than 100 images with multiple batches, I can do the following code. Otherwise, manually split to train/validate/test
# #determine how many batches of data are available in the validation set using cardinality:
# val_batches = tf.data.experimental.cardinality(validation_dataset)
# # val_batches.numpy() #32 batches to accomodate all 1000 validation instances
# test_dataset = validation_dataset.take(val_batches // 5) #take first 20% of the validation batches and save as test, size is 6
# validation_dataset = validation_dataset.skip(val_batches // 5) #skips first 20%, size is 26
# print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset)) #26
# print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset)) #6


#tutorial using images for customizing the pretrained model:
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
BATCH_SIZE = 32 #gradient and neural network parameters updated after each iteration of 32
IMG_SIZE = (160, 160) #ensure a uniform resizing since input images are all diff sizes
train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE) #Found 2000 files belonging to 2 classes.
validation_dataset = image_dataset_from_directory(validation_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE) #Found 1000 files belonging to 2 classes.
#show first nine images and labels from training set:
class_names = train_dataset.class_names #extract class names previous function inferred from subdir's
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1): #load first iteration batch from training dataset
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1) #setup axis on a 3x3 grid
    plt.imshow(images[i].numpy().astype("uint8")) #plot each image
    plt.title(class_names[labels[i]]) #output associated label for chosen image
    plt.axis("off")
#determine how many batches of data are available in the validation set using cardinality:
val_batches = tf.data.experimental.cardinality(validation_dataset)
#val_batches.numpy() #32 batches to accomodate all 1000 validation instances
test_dataset = validation_dataset.take(val_batches // 5) #take first 20% of the validation batches and save as test, size is 6
validation_dataset = validation_dataset.skip(val_batches // 5) #skips first 20%, size is 26
print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset)) #26
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset)) #6
