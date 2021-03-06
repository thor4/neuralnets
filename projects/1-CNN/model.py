## CNN model to dissociate confidence from accuracy
#use transfer learning tutorial here: https://www.tensorflow.org/tutorials/images/transfer_learning

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory

tf.__version__ #2.4.0

#clear workspace variables in iPython:
%reset 


def init_sets():
    curr_dir = os.getcwd() #make sure I'm in CNN project folder
    # '/workspaces/neuralnets/projects/1-CNN' (equiv to PATH var below)
    # train_dir = os.path.join(curr_dir, 'images/train')
    train_dir = os.path.join(curr_dir, 'images/train_range') #range of tilts/contrasts
    # '/workspaces/neuralnets/projects/1-CNN/images/train'
    # there is a 'clock' and 'cclock' folder in each of these folders (train/test/validate)
    # validation_dir = os.path.join(curr_dir, 'images/validation')
    validation_dir = os.path.join(curr_dir, 'images/validation_range') #range of tilts/contrasts
    # '/workspaces/neuralnets/projects/1-CNN/images/validation' 
    # test_dir = os.path.join(curr_dir, 'images/test')
    test_dir = os.path.join(curr_dir, 'images/test_range') #range of tilts/contrasts
    # '/workspaces/neuralnets/projects/1-CNN/images/validation'
    BATCH_SIZE = 32 #stick with one iteration for training and validation per optimal batch guidance here:
    # https://ai.stackexchange.com/questions/8560/how-do-i-choose-the-optimal-batch-size
    # practically, this means only one update of gradient and neural network parameters
    IMG_SIZE = (160, 160) #forces a resize from 170x170 since MobileNetV2 has weights only for certain sizes
    train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE) #Found 120 files belonging to 2 classes.
                                             #13332 images (10k/20k tot db), 11988 (range)
    validation_dataset = image_dataset_from_directory(validation_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE) #Found 40 files belonging to 2 classes.
                                                  #5334 images (10k/20k tot db), 4788 (range)
    test_dataset = image_dataset_from_directory(test_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE) #Found 40 files belonging to 2 classes.
                                                  #1334 images (10k/20k tot db), 1224 (range)
    class_names = train_dataset.class_names #extract class names previous function inferred from subdir's
    return BATCH_SIZE,IMG_SIZE,train_dataset,validation_dataset,test_dataset,class_names

BATCH_SIZE, IMG_SIZE, train_dataset, validation_dataset, test_dataset, class_names = init_sets()

plt.figure(figsize=(10, 10)) #show first nine images and labels from training set
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
def prefetch(train_dataset, validation_dataset, test_dataset):
    AUTOTUNE = tf.data.AUTOTUNE #prompts the tf.data runtime to tune the value dynamically at runtime
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    return train_dataset,validation_dataset,test_dataset,AUTOTUNE

train_dataset, validation_dataset, test_dataset, AUTOTUNE = prefetch(train_dataset, validation_dataset, test_dataset)

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
#no need to save the range model since it's only 53.74% accurate

model = tf.keras.models.load_model('models/10kim_1con') #load previously trained model 1 (not fine-tuned)
#after loading, the convolutional base model weights need to be frozen:
model.get_layer(name='mobilenetv2_1.00_160').trainable=False #get mobilenet base then freeze it
model.summary() #verify architecture, trainable: 1,281 params between last two layers

model.get_layer(name='dense').input_shape #gets last layer to confirm input (None,1280) features
model.get_layer(name='dense').output_shape #confirm output is a single logit

#now time for fine-tuning the model where we train the weights of the top layers of the conv base model concurrently with the classifier
#The goal of fine-tuning is to adapt specialized features found in the highest layers to work 
#with the new dataset, rather than overwrite the generic learning found in the lowest layers
#make sure the classifier is trained on the new dataset first. otherwise if you try with randomly initialized weights in the classifier, 
#the gradient updates will be too large and the pre-trained conv model will forget what it learned
model.get_layer(name='mobilenetv2_1.00_160').trainable=True #first, unfreeze the base convolutional model
print("Number of layers in the base model: ", len(model.get_layer(name='mobilenetv2_1.00_160').layers)) #how many layers are in the base model (154)
fine_tune_at = 100 #fine-tune from this layer onwards
for layer in model.get_layer(name='mobilenetv2_1.00_160').layers[:fine_tune_at]: # freeze all the layers before the `fine_tune_at` layer
  layer.trainable =  False
base_learning_rate = 0.0001 #define the learning rate
#we have set the bottom layers (1-99 inclusive) to be untrainable. now recompile so changes take effect
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10), #use lower rate since model is much larger and want to avoid overfitting
              metrics=['accuracy'])
model.summary() #now we have 1,862,721 trainable parameters
len(model.trainable_variables) #this includes 56 total variables
fine_tune_epochs = 10 #define additional epochs for fine_tuning
#total_epochs =  initial_epochs + fine_tune_epochs
history_fine = model.fit(train_dataset,
                         #epochs=total_epochs,
                         epochs=fine_tune_epochs,
                         #initial_epoch=history.epoch[-1], #start from last epoch
                         validation_data=validation_dataset)
#if the validation loss is much higher than the training loss, you may get some overfitting
#also get some overfitting if the new training set is relatively small and similar to the original MobileNet V2 datasets
# acc += history_fine.history['accuracy'] #extract and append fine_tune history of accuracy scores on training set as list
# val_acc += history_fine.history['val_accuracy'] #same as above but for validation set
# loss += history_fine.history['loss'] #same as above but loss instead of acc
# val_loss += history_fine.history['val_loss'] #same as above
# plt.figure(figsize=(8, 8))
# plt.subplot(2, 1, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.ylim([0.8, 1]) #zoom in on one section of plot
# plt.plot([initial_epochs-1,initial_epochs-1],
#           plt.ylim(), label='Start Fine Tuning')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(2, 1, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.ylim([0, 1.0])
# plt.plot([initial_epochs-1,initial_epochs-1],
#          plt.ylim(), label='Start Fine Tuning')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.xlabel('epoch')
# plt.show() #~98% accuracy on validation set
acc = history_fine.history['accuracy'] #extract and store history of accuracy scores on training set as list
val_acc = history_fine.history['val_accuracy'] #extract and store history of accuracy scores on validation set as list
loss = history_fine.history['loss'] #extract and store history of loss scores on training set as list
val_loss = history_fine.history['val_loss'] #extract and store history of losss scores on validation set as list
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
loss, accuracy = model.evaluate(test_dataset) #now test the model's performance on the test set
print('Test accuracy :', accuracy) #100% accuracy
model.save('models/10kim_1con_ft') #save fine-tuned model 1 into the 10kim 1con ft folder
#contrasts = [.3, .45, 1] & tilts = [.1, .2, .4, .8, 1.6, 3.2]:
model.save('models/18kim_range_ft') #save fine-tuned model 2 into the 18kim range ft folder (ft=fine-tuned)

#model = tf.keras.models.load_model('models/10kim_1con_ft') #load previously trained fine-tuned model
model = tf.keras.models.load_model('models/18kim_range_ft') #load previously trained fine-tuned range model (model 2)
model.summary() #verify architecture, trainable: 2,225,153 params between last 100 layers

#plot images with predictions from test dataset
plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].astype("uint8")) #plot from test set
  plt.title(class_names[predictions[i]]) #apply labels from prediction set
  plt.axis("off")

#first, get confidence and accuracy on a single batch:
#calculate confidence on logits
def get_conf_logits(test_dataset, model):
    #Retrieve a batch of images from the test set
    image_batch, label_batch = test_dataset.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch).flatten() #run batch through model and return logits
    plt.hist(np.array(predictions)) #fine-tune: [-30,30] thres: 20, non ft: [-5,5] thres: 2, 0.5 for range
    logit_thres_min = -0.5 #set min logit conf threshold for class 0
    logit_thres_max = 0.5 #set max logit conf threshold for class 1
    threshold = tf.math.logical_or(predictions < logit_thres_min, predictions > logit_thres_max) #set conf threshold
    confidence = tf.where(threshold, 1, 0) #low confidence is 0, high confidence is 1
    high_conf_avg = tf.math.reduce_mean(tf.dtypes.cast(confidence, tf.float64)) #avg of high conf
    print('High Conf Avg:', high_conf_avg.numpy()) #verify the avg high conf is within reason
    return label_batch,predictions,logit_thres_min,logit_thres_max
#calculate confidence on sigmoid-transformed logits
def get_conf_sigtrans(predictions):
    predictions_sigtrans = tf.nn.sigmoid(predictions) #apply sigmoid activation function to transform logits to [0,1]
    plt.hist(np.array(predictions_sigtrans)) #visualize the sigmoid-transformed logits
    threshold2 = tf.math.logical_or(predictions_sigtrans < 0.25, predictions_sigtrans > 0.75) #set conf threshold based on sigmoid-transformed logits, closer to 0 or 1 to track either class
    confidence2 = tf.where(threshold2, 1, 0) #low confidence is 0, high confidence is 1 based on sig-trans logits
    high_conf_avg2 = tf.math.reduce_mean(tf.dtypes.cast(confidence2, tf.float64)) #avg of high conf based on sig-trans logits
    print('High Conf Avg:', high_conf_avg2.numpy()) #verify the avg high conf is within reason
    return predictions_sigtrans
def get_acc(label_batch, predictions_sigtrans):
    predictions_sigtrans = tf.where(predictions_sigtrans < 0.5, 0, 1) #round down or up accordingly since it's a binary classifier
    accuracy = tf.where(tf.equal(predictions_sigtrans,label_batch),1,0) #correct is 1 and incorrect is 0
    print('Predictions:\n', predictions_sigtrans.numpy())
    print('Labels:\n', label_batch) #nice job predicting
    print('Incorrect:\n', accuracy.numpy())
    avg_accuracy = tf.math.reduce_mean(tf.dtypes.cast(accuracy, tf.float64)) #avg of accuracy to get performance
    print('Accuracy:', avg_accuracy.numpy()) #base: 0.965 vs 0.9528 with model.evaluate (maybe the sigmoid or rounding steps are diff)
def thres_sigtrans(logit_thres_min, logit_thres_max):
    thres = tf.constant([logit_thres_min, logit_thres_max])
    thres_sig = tf.nn.sigmoid(thres) #-0.5 logit sigmoid-transforms to 0.3775407 and 0.5 sigmoid-transforms to 0.6224593
    print(thres[0].numpy(),'logit sigmoid-transforms to', thres_sig[0].numpy(),'and', thres[1].numpy(),'logit sigmoid-transforms to', thres_sig[1].numpy())

label_batch, predictions, logit_thres_min, logit_thres_max = get_conf_logits(test_dataset, model)
predictions_sigtrans = get_conf_sigtrans(predictions)
get_acc(label_batch, predictions_sigtrans)

#next, get confidence and accuracy on a total dataset:

def get_conf_acc_dataset(test_dataset, model, logit_thres_min, logit_thres_max):
    all_conf=tf.zeros([], tf.float64) #initialize array to hold all confidence ratings (single element)
    all_pred=tf.zeros([], tf.float64) #initialize array to hold all prediction logits (single element)
    all_labels=tf.zeros([], tf.float64) #initialize array to hold all actual labels (single element)
    all_acc=tf.zeros([], tf.float64) #initialize array to hold all accuracy indicators (single element)
    all_avg_acc=tf.zeros([], tf.float64) #initialize array to hold all avg accuracy indicators (single element)
    all_pred_sigtrans=tf.zeros([], tf.float64) #initialize array to hold all avg accuracy indicators (single element)

    for image_batch, label_batch in test_dataset.as_numpy_iterator():
        predictions = model.predict_on_batch(image_batch).flatten() #run batch through model and return logits
        threshold = tf.math.logical_or(predictions < logit_thres_min, predictions > logit_thres_max) #set conf threshold at -20 and 20
        confidence = tf.where(threshold, 1, 0) #low confidence is 0, high confidence is 1
        all_conf = tf.experimental.numpy.append(all_conf, confidence)
        all_pred = tf.experimental.numpy.append(all_pred, predictions)
        all_labels = tf.experimental.numpy.append(all_labels, label_batch)
        predictions = tf.nn.sigmoid(predictions) #apply sigmoid activation function to transform logits to [0,1]
        predictions = tf.where(predictions < 0.5, 0, 1) #round down or up accordingly since it's a binary classifier
        all_pred_sigtrans = tf.experimental.numpy.append(all_pred_sigtrans, predictions)
        accuracy = tf.where(tf.equal(predictions,label_batch),1,0) #correct is 1 and incorrect is 0
        all_acc = tf.experimental.numpy.append(all_acc, accuracy)
        if all_acc.size==BATCH_SIZE+1: #accumulating avg loop, check to see if on first loop
            avg_accuracy = tf.reduce_mean(tf.dtypes.cast(accuracy, tf.float64)) #avg performance, init avg acc of first batch
        else:
            #average last/running avg with current batch's avg:
            avg_accuracy = tf.reduce_mean(tf.concat([avg_accuracy,tf.reduce_mean(tf.dtypes.cast(accuracy, tf.float64))],0))
        #save current accuracy in array
        all_avg_acc = tf.experimental.numpy.append(all_avg_acc, tf.reduce_mean(tf.dtypes.cast(accuracy, tf.float64)))
    #tf.size(all_conf) #1335 elements, 1334 images + 1 placeholder 0 at beginning
    all_conf = all_conf[1:]
    all_pred = all_pred[1:]
    all_pred_sigtrans = all_pred_sigtrans[1:]
    all_acc = all_acc[1:]  
    all_labels = all_labels[1:]
    all_avg_acc = all_avg_acc[1:] #drop first placeholder element
    high_conf_avg = tf.reduce_mean(all_conf) #avg of high conf
    avg_acc = tf.reduce_mean(all_acc) #avg performance
    batch_avg_acc = tf.reduce_mean(all_avg_acc) #avg avg performance on each batch

    print('Predictions:\n', all_pred_sigtrans[:])
    print('Labels:\n', all_labels) #nice job predicting
    print('Incorrect:\n', all_acc)
    print('High Confidence:', high_conf_avg.numpy()) #base: 0.503
    print('Accuracy:', avg_acc.numpy()) #base: 0.965 vs 0.9528 with model.evaluate (maybe the sigmoid or rounding steps are diff)
    print('Batch Accuracy:', batch_avg_acc.numpy()) #overall avg of each batches' avg performance, same as avg_acc
    print('Accumulating Avg Accuracy:', avg_accuracy.numpy())
    return all_avg_acc,avg_accuracy

all_avg_acc, avg_accuracy = get_conf_acc_dataset(test_dataset, model, logit_thres_min, logit_thres_max)

#sigmoid transform 0.5 and -0.5 to see what they correspond to in the sigmoid transform space
thres_sigtrans(logit_thres_min, logit_thres_max)

i=0
for i in range(all_avg_acc.size-1): #go through entire array and calculate the accumulating avg
  if i==0:
    running_avg = tf.reduce_mean(all_avg_acc[0:i+2])
  else:
    running_avg = tf.reduce_mean(tf.concat([running_avg,all_avg_acc[i+1]],0))

print('running average is', running_avg.numpy())
print('one-shot average is', tf.reduce_mean(all_avg_acc).numpy())

#toy example of accumulating average:
test = tf.constant([1, 2, 3, 4, 5], dtype=tf.float64)
for i in range(5-1): #go through entire array and calculate the accumulating avg
  if i==0:
    running_avg = tf.reduce_mean(test[0:i+2])
  else:
    running_avg = tf.reduce_mean(tf.concat([running_avg,test[i+1]],0))

print('running average is', running_avg.numpy())
print('one-shot average is', tf.reduce_mean(test).numpy())


#stackoverflow/github issues question
current_set = set17 #define set to process.
all_acc=tf.zeros([], tf.float64) #initialize array to hold all accuracy indicators (single element)
loss, acc = model.evaluate(current_set) #now test the model's performance on the test set
for image_batch, label_batch in current_set.as_numpy_iterator():
    predictions = model.predict_on_batch(image_batch).flatten() #run batch through model and return logits
    predictions = tf.nn.sigmoid(predictions) #apply sigmoid activation function to transform logits to [0,1]
    predictions = tf.where(predictions < 0.5, 0, 1) #round down or up accordingly since it's a binary classifier
    accuracy = tf.where(tf.equal(predictions,label_batch),1,0) #correct is 1 and incorrect is 0
    all_acc = tf.experimental.numpy.append(all_acc, accuracy)
all_acc = all_acc[1:]  #drop first placeholder element
avg_acc = tf.reduce_mean(all_acc)
print('My Accuracy:', avg_acc.numpy()) 
print('Tf Accuracy:', acc) 


#next, generate images with different combinations of contrast + tilt 3x3
#create dataset for each and run through loop to get avg high/low conf + acc for non-finetuned model trained on 20k 1 con, 2.26 tilt images
#export to table for doby

#1334 images w tilt 2.26, contrast 1:
#High Confidence: 0.503, Low Confidence: 0.497, Accuracy: 0.965

#Model 1 testing:
def model1_init_sets(BATCH_SIZE, IMG_SIZE, AUTOTUNE):
    curr_dir = os.getcwd() #make sure I'm in CNN project folder
    set1_dir = os.path.join(curr_dir, 'images/set1-t_2.26-c_1')
    set2_dir = os.path.join(curr_dir, 'images/set2-t_2.26-c_0.3')
    set3_dir = os.path.join(curr_dir, 'images/set3-t_2.26-c_0.45')
    set4_dir = os.path.join(curr_dir, 'images/set4-t_1.13-c_1')
    set5_dir = os.path.join(curr_dir, 'images/set5-t_1.13-c_0.3')
    set6_dir = os.path.join(curr_dir, 'images/set6-t_1.13-c_0.45')
    set7_dir = os.path.join(curr_dir, 'images/set7-t_4.52-c_1')
    set8_dir = os.path.join(curr_dir, 'images/set8-t_4.52-c_0.3')
    set9_dir = os.path.join(curr_dir, 'images/set9-t_4.52-c_0.45')
    set1 = image_dataset_from_directory(set1_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    set2 = image_dataset_from_directory(set2_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    set3 = image_dataset_from_directory(set3_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    set4 = image_dataset_from_directory(set4_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    set5 = image_dataset_from_directory(set5_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    set6 = image_dataset_from_directory(set6_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    set7 = image_dataset_from_directory(set7_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    set8 = image_dataset_from_directory(set8_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    set9 = image_dataset_from_directory(set9_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    set1 = set1.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    set2 = set2.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    set3 = set3.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    set4 = set4.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    set5 = set5.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    set6 = set6.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    set7 = set7.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    set8 = set8.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    set9 = set9.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    return set1,set2,set3,set4,set5,set6,set7,set8,set9

set1,set2,set3,set4,set5,set6,set7,set8,set9 = model1_init_sets(BATCH_SIZE, IMG_SIZE, AUTOTUNE)

#Model 2 testing:
def model2_init_sets(BATCH_SIZE, IMG_SIZE, AUTOTUNE):
    curr_dir = os.getcwd() #make sure I'm in CNN project folder
    set1_dir = os.path.join(curr_dir, 'images/datasets/model2/s1-t_0.1-c_0.3')
    set2_dir = os.path.join(curr_dir, 'images/datasets/model2/s2-t_0.1-c_0.45')
    set3_dir = os.path.join(curr_dir, 'images/datasets/model2/s3-t_0.1-c_1')
    set4_dir = os.path.join(curr_dir, 'images/datasets/model2/s4-t_0.2-c_0.3')
    set5_dir = os.path.join(curr_dir, 'images/datasets/model2/s5-t_0.2-c_0.45')
    set6_dir = os.path.join(curr_dir, 'images/datasets/model2/s6-t_0.2-c_1')
    set7_dir = os.path.join(curr_dir, 'images/datasets/model2/s7-t_0.4-c_0.3')
    set8_dir = os.path.join(curr_dir, 'images/datasets/model2/s8-t_0.4-c_0.45')
    set9_dir = os.path.join(curr_dir, 'images/datasets/model2/s9-t_0.4-c_1')
    set10_dir = os.path.join(curr_dir, 'images/datasets/model2/s10-t_0.8-c_0.3')
    set11_dir = os.path.join(curr_dir, 'images/datasets/model2/s11-t_0.8-c_0.45')
    set12_dir = os.path.join(curr_dir, 'images/datasets/model2/s12-t_0.8-c_1')
    set13_dir = os.path.join(curr_dir, 'images/datasets/model2/s13-t_1.6-c_0.3')
    set14_dir = os.path.join(curr_dir, 'images/datasets/model2/s14-t_1.6-c_0.45')
    set15_dir = os.path.join(curr_dir, 'images/datasets/model2/s15-t_1.6-c_1')
    set16_dir = os.path.join(curr_dir, 'images/datasets/model2/s16-t_3.2-c_0.3')
    set17_dir = os.path.join(curr_dir, 'images/datasets/model2/s17-t_3.2-c_0.45')
    set18_dir = os.path.join(curr_dir, 'images/datasets/model2/s18-t_3.2-c_1')
    set1 = image_dataset_from_directory(set1_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE) #3000 images 2 classes
    set2 = image_dataset_from_directory(set2_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    set3 = image_dataset_from_directory(set3_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    set4 = image_dataset_from_directory(set4_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    set5 = image_dataset_from_directory(set5_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    set6 = image_dataset_from_directory(set6_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    set7 = image_dataset_from_directory(set7_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE) #3000 images 2 classes
    set8 = image_dataset_from_directory(set8_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    set9 = image_dataset_from_directory(set9_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    set10 = image_dataset_from_directory(set10_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    set11 = image_dataset_from_directory(set11_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    set12 = image_dataset_from_directory(set12_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    set13 = image_dataset_from_directory(set13_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE) #3000 images 2 classes
    set14 = image_dataset_from_directory(set14_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    set15 = image_dataset_from_directory(set15_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    set16 = image_dataset_from_directory(set16_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    set17 = image_dataset_from_directory(set17_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    set18 = image_dataset_from_directory(set18_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    set1 = set1.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    set2 = set2.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    set3 = set3.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    set4 = set4.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    set5 = set5.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    set6 = set6.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    set7 = set7.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    set8 = set8.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    set9 = set9.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    set10 = set10.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    set11 = set11.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    set12 = set12.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    set13 = set13.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    set14 = set14.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    set15 = set15.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    set16 = set16.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    set17 = set17.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    set18 = set18.prefetch(buffer_size=AUTOTUNE) #will prefetch an optimal number of batches
    return set1,set2,set3,set4,set5,set6,set7,set8,set9,set10,set11,set12,set13,set14,set15,set16,set17,set18

set1,set2,set3,set4,set5,set6,set7,set8,set9,set10,set11,set12,set13,set14,set15,set16,set17,set18 = model2_init_sets(BATCH_SIZE, IMG_SIZE, AUTOTUNE)

current_set = set18 #define set to process. must do all sets, one at a time
#eps = sys.float_info.epsilon

#STOPPED HERE - work on extracting accumulating average for previous results, then run with new confidence threshold to see if effect holds
def model_tests(model, current_set):
    all_conf=tf.zeros([], tf.int32) #initialize array to hold all confidence ratings (single element)
    all_acc=tf.zeros([], tf.int32) #initialize array to hold all accuracy indicators (single element)
    loss, acc = model.evaluate(current_set) #now test the model's performance on the test set
    for image_batch, label_batch in current_set.as_numpy_iterator():
        predictions = model.predict_on_batch(image_batch).flatten() #run batch through model and return logits
        threshold = tf.math.logical_or(predictions < -0.5, predictions > 0.5) #set conf threshold at -20 and 20
        confidence = tf.where(threshold, 1, 0) #low confidence is 0, high confidence is 1
        all_conf = tf.experimental.numpy.append(all_conf, confidence)
    #all_pred = tf.experimental.numpy.append(all_pred, predictions)
        predictions = tf.nn.sigmoid(predictions) #apply sigmoid activation function to transform logits to [0,1]
        predictions = tf.where(predictions < 0.5, 0, 1) #round down or up accordingly since it's a binary classifier
        accuracy = tf.where(tf.equal(predictions,label_batch),1,0) #correct is 1 and incorrect is 0
        all_acc = tf.experimental.numpy.append(all_acc, accuracy)
    # temp_avg_acc = tf.math.reduce_mean(tf.dtypes.cast(accuracy, tf.float16)) #avg of accuracy
    # new_acc = 
    all_conf = all_conf[1:]
#all_pred = all_pred[1:]
    all_acc = all_acc[1:]  #drop first placeholder element
    high_conf_avg = tf.math.reduce_mean(tf.dtypes.cast(all_conf, tf.float16)) #avg of high conf
#low_conf_avg = 1 - high_conf_avg
    avg_acc = tf.math.reduce_mean(tf.dtypes.cast(all_acc, tf.float16)) #avg of accuracy
    print('High Confidence:', high_conf_avg.numpy()) #base: 0.503
#print('Low Confidence:', low_conf_avg.numpy()) #base: 0.497
    print('My Accuracy:', avg_acc.numpy()) #base: 0.965 vs 0.9528 with model.evaluate (maybe the sigmoid or rounding steps are diff)
    print('Tf Accuracy:', acc) #base: 0.965 vs 0.9528 with model.evaluate (maybe the sigmoid or rounding steps are diff)

# model.evaluate docs (find more incl TF's):
# https://keras.io/api/models/model_training_apis/
    # https://keras.rstudio.com/reference/evaluate.html

model_tests(model, current_set)