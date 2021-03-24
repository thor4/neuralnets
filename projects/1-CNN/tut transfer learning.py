# Transfer learning and fine-tuning tutorial
# https://www.tensorflow.org/tutorials/images/transfer_learning

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory

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
BATCH_SIZE = 32 #stick with one iteration for training and validation per optimal batch guidance here:
# https://ai.stackexchange.com/questions/8560/how-do-i-choose-the-optimal-batch-size
# practically, this means only one update of gradient and neural network parameters
IMG_SIZE = (160, 160) #forces a resize from 170x170 since MobileNetV2 has weights only for certain sizes
train_dataset = image_dataset_from_directory(train_dir,
                                             #color_mode="grayscale", #rgb by default, save 1 chan instead of 3
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE) #Found 120 files belonging to 2 classes.
                                             #now its 132 files, 13332
validation_dataset = image_dataset_from_directory(validation_dir,
                                                  #color_mode="grayscale", #rgb by default, save 1 chan instead of 3
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE) #Found 40 files belonging to 2 classes.
                                                  #now its 54 files, 5334
test_dataset = image_dataset_from_directory(test_dir,
                                                  #color_mode="grayscale", #rgb by default, save 1 chan instead of 3
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE) #Found 40 files belonging to 2 classes.
                                                  #now its 14 files, 1334
#show first nine images and labels from training set:
class_names = train_dataset.class_names #extract class names previous function inferred from subdir's
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1): #load first iteration batch from training dataset
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1) #setup axis on a 3x3 grid
    plt.imshow(images[i].numpy().astype("uint8"),cmap='gray') #plot each image
    plt.title(class_names[labels[i]]) #output associated label for chosen image
    plt.axis("off")
#if I need to use more than 100 images with more than 5 batches, I can do the following code. Otherwise, manually split to train/validate/test
# #determine how many batches of data are available in the validation set using cardinality:
# val_batches = tf.data.experimental.cardinality(validation_dataset)
# # val_batches.numpy() #32 batches to accomodate all 1000 validation instances
# test_dataset = validation_dataset.take(val_batches // 5) #take first 20% of the validation batches and save as test, size is 6
# validation_dataset = validation_dataset.skip(val_batches // 5) #skips first 20%, size is 26
# print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset)) #26
# print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset)) #6
#now configure the dataset for performance using buffered prefetching to load images from disk without having I/O become blocking
#may not need the following until we do more than one batch at a time:
AUTOTUNE = tf.data.AUTOTUNE #prompts the tf.data runtime to tune the value dynamically at runtime
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
#MobileNetV2 expects images with pixel values [-1,1],  but our images are [0,255]. Need to rescale:
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input #this pre-processing method rescales input according to what MobileNetv2 expects
#Now create the base model MobileNetV2 with pre-loaded weights trained on ImageNet
IMG_SHAPE = IMG_SIZE + (3,) #adds a third dimension of 3 to hold color channels ie: (160,160,3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, #needs one of these sizes: [96, 128, 160, 192, 224]
                                               include_top=False, #ensures classification layers at the top are not loaded, ideal for feature extraction
                                               weights='imagenet') #has weights for each preferred size
#This feature extractor converts each 160x160x3 image into a 5x5x1280 block of features. 
# Let's see what it does to an example batch of images:
image_batch, label_batch = next(iter(train_dataset)) #image_batch= (32,170,170,1), label_batch=(32,), 32 images
feature_batch = base_model(image_batch) #extract features from one batch of images
print(feature_batch.shape) #(32,5,5,1280) reduces total overal dimensionality of input images
#the base_model is now considered the convolutional base model. we will freeze its layers so the weights do not change during training
base_model.trainable = False #freezes entire model so no weights get updated
# Let's take a look at the base model architecture
base_model.summary() #Total params: 2,257,984, all non-trainable
global_average_layer = tf.keras.layers.GlobalAveragePooling2D() #create layer to avg over the middle 5x5 dimensions leaving 2d
feature_batch_average = global_average_layer(feature_batch) #avg over 5,5 leaving 32,1280
print(feature_batch_average.shape) #(32,1280)
#now create a layer that will generate a single prediction per image using logits. positive means class 1, neg class 2
prediction_layer = tf.keras.layers.Dense(1) #create a densely-connected NN layer with a single logit output prediction
prediction_batch = prediction_layer(feature_batch_average) #generate predictions for first batch of training images
print(prediction_batch.shape) #(32,1) 32 logits predicting clockwise or counterclockwise
#now time to build a new model by chaining together the rescaling, base_model and feature extractors using the Keras Functional API
inputs = tf.keras.Input(shape=(160, 160, 3)) #ensure input size is one of the predefined MobileNetV2 requires
x = preprocess_input(inputs) #rescale the images to [-1,1]
x = base_model(x, training=False) #process images through convolutional base with training=False to ensure batchNorm layer weights stay locked
x = global_average_layer(x) #extract features by avg over spatial 5x5 dim's leaving (32,1280)
x = tf.keras.layers.Dropout(0.2)(x) #add dropout layer which randomly drop 20% of the input to prevent overfitting during training (turns some nodes off)
outputs = prediction_layer(x) #define output as single prediction
model = tf.keras.Model(inputs, outputs) #build the model
base_learning_rate = 0.0001 #define the learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate), #now compile before training
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), #use this loss since model provides a linear output
              metrics=['accuracy'])
model.summary() #Total params: 2,259,265, Trainable params: 1,281, Non-trainable params: 2,257,984
#the only trainable parameters are in the last layer between the global pooling 1280 node (which has dropout applied to it) and the dense single-node prediction layer
len(model.trainable_variables) #the 1281 trainable parameters are divided between 2 tf.variables: weights (1280) and biases (1)
initial_epochs = 10 #define training epochs
loss0, accuracy0 = model.evaluate(validation_dataset) #obtain initial performance on 2 validation batches
print("initial loss: {:.2f}".format(loss0)) #0.72 (100 images, con 0.45), 1.37 (100 im, con 1), 1.04 (10k im, con1)
print("initial accuracy: {:.2f}".format(accuracy0)) #0.47 (100 im, con 0.45), 0.5 (100 im, con 1), 0.5 (10k im, con1)
history = model.fit(train_dataset, #trains the model using the training dataset
                    epochs=initial_epochs, #runs specified number of iterations over all training batches
                    validation_data=validation_dataset) #tests against validation dataset after each iteration
#now visualize results of using the MobileNetV2 base model as a fixed feature extractor
acc = history.history['accuracy'] #extract and store history of accuracy scores on training set as list
val_acc = history.history['val_accuracy'] #extract and store history of accuracy scores on validation set as list
loss = history.history['loss'] #extract and store history of loss scores on training set as list
val_loss = history.history['val_loss'] #extract and store history of losss scores on validation set as list
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show() #accuracy trends up over time and the loss goes down
# If you are wondering why the validation metrics are clearly better than the training metrics, the main factor is because 
# layers like tf.keras.layers.BatchNormalization and tf.keras.layers.Dropout affect accuracy during training. 
# They are turned off when calculating validation loss. To a lesser extent, it is also because training metrics report the 
# average for an epoch, while validation metrics are evaluated after the epoch, so validation metrics see a model that has trained slightly longer.
# Save the entire model as a SavedModel.
model.save('models/10kim_1con') #save the trained model into the 10kim 1con folder





#tutorial using images for customizing the pretrained model:
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
#now configure the dataset for performance using buffered prefetching to load images from disk without having I/O become blocking
AUTOTUNE = tf.data.AUTOTUNE #prompts the tf.data runtime to tune the value dynamically at runtime
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
#now time to use data augmentation for tutorial purposes only, not needed on gabors
data_augmentation = tf.keras.Sequential([ #define a keras model with 2 layers: horizontal and rotational flips
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
]) #These layers are active only during training, when you call model.fit. Inactive when useed in inference mode in model.evaulate or model.fit.
for image, _ in train_dataset.take(1):
  plt.figure(figsize=(10, 10))
  first_image = image[0]
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    #expand to make 4d shape expected by RandomFlip/Rotation (samples, height, width, channels):
    augmented_image = data_augmentation(tf.expand_dims(first_image, 0)) 
    plt.imshow(augmented_image[0] / 255)
    plt.axis('off')

#MobileNetV2 expects images with pixel values [-1,1],  but our images are [0,255]. Need to rescale:
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input #this pre-processing method rescales input according to what MobileNetv2 expects
#or could rescale pixel values from [0,255] to [-1, 1] using a Rescaling layer:
#rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)
#docs for above: https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/Rescaling
#when using other tf.keras.applications, be sure to check the API doc to determine if they expect pixels in [-1,1] or [0,1], or use the included preprocess_input function.
#Now create the base model MobileNetV2 with pre-loaded weights trained on ImageNet
IMG_SHAPE = IMG_SIZE + (3,) #adds a third dimension of 3 to hold color channels ie: (160,160,3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False, #ensures classification layers at the top are not loaded, ideal for feature extraction
                                               weights='imagenet')
#This feature extractor converts each 160x160x3 image into a 5x5x1280 block of features. 
# Let's see what it does to an example batch of images:
image_batch, label_batch = next(iter(train_dataset)) #image_batch= (32,160,160,3), label_batch=(32,), 32 images
feature_batch = base_model(image_batch) #extract features from one batch of images
print(feature_batch.shape) #(32,5,5,1280) reduces total overal dimensionality of input images
#the base_model is now considered the convolutional base model. we will freeze its layers so the weights do not change during training
base_model.trainable = False #freezes entire model so no weights get updated
# Let's take a look at the base model architecture
base_model.summary() #Total params: 2,257,984
global_average_layer = tf.keras.layers.GlobalAveragePooling2D() #create layer to avg over the middle 5x5 dimensions leaving 2d
feature_batch_average = global_average_layer(feature_batch) #avg over 5,5 leaving 32,1280
print(feature_batch_average.shape) #(32,1280)
#now create a layer that will generate a single prediction per image using logits. positive means class 1, neg class 2
prediction_layer = tf.keras.layers.Dense(1) #create a densely-connected NN layer with a single logit output prediction
prediction_batch = prediction_layer(feature_batch_average) #generate predictions for first batch of training images
print(prediction_batch.shape) #(32,1) 32 logits predicting cat or dog
#now time to build a new model by chaining together the data augmentation, rescaling, base_model and feature extractors using the Keras Functional API
inputs = tf.keras.Input(shape=(160, 160, 3)) #ensure input size is one of the predefined MobileNetV2 requires
x = data_augmentation(inputs) #flip/rotate to increase images
x = preprocess_input(x) #rescale the images to [-1,1]
x = base_model(x, training=False) #process images through convolutional base with training=False to ensure batchNorm layer weights stay locked
x = global_average_layer(x) #extract features by avg over spatial 5x5 dim's leaving (32,1280)
x = tf.keras.layers.Dropout(0.2)(x) #add dropout layer which randomly drop 20% of the input to prevent overfitting during training (turns some nodes off)
outputs = prediction_layer(x) #define output as single prediction
model = tf.keras.Model(inputs, outputs) #build the model
base_learning_rate = 0.0001 #define the learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate), #now compile before training
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), #use this loss since model provides a linear output
              metrics=['accuracy'])
model.summary() #Total params: 2,259,265, Trainable params: 1,281, Non-trainable params: 2,257,984
#the only trainable parameters are in the last layer between the global pooling 1280 node (which has dropout applied to it) and the dense single-node prediction layer
len(model.trainable_variables) #the 1281 trainable parameters are divided between 2 tf.variables: weights (1280) and biases (1)
initial_epochs = 10 #define training epochs
loss0, accuracy0 = model.evaluate(validation_dataset) #obtain initial performance on 26 validation batches
print("initial loss: {:.2f}".format(loss0)) #0.61
print("initial accuracy: {:.2f}".format(accuracy0)) #0.60
history = model.fit(train_dataset, #trains the model using the training dataset
                    epochs=initial_epochs, #runs specified number of iterations over all training batches
                    validation_data=validation_dataset) #tests against validation dataset after each iteration
#now visualize results of using the MobileNetV2 base model as a fixed feature extractor
acc = history.history['accuracy'] #extract and store history of accuracy scores on training set as list
val_acc = history.history['val_accuracy'] #extract and store history of accuracy scores on validation set as list
loss = history.history['loss'] #extract and store history of loss scores on training set as list
val_loss = history.history['val_loss'] #extract and store history of losss scores on validation set as list
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show() #accuracy trends up over time and the loss goes down
# If you are wondering why the validation metrics are clearly better than the training metrics, the main factor is because 
# layers like tf.keras.layers.BatchNormalization and tf.keras.layers.Dropout affect accuracy during training. 
# They are turned off when calculating validation loss. To a lesser extent, it is also because training metrics report the 
# average for an epoch, while validation metrics are evaluated after the epoch, so validation metrics see a model that has trained slightly longer.