# TensorFlow 2 quickstart for beginners

import tensorflow as tf

tf.__version__ #2.4.0

mnist = tf.keras.datasets.mnist #create handle to LeCun's MNIST dataset

(X_train, Y_train), (X_test, Y_test) = mnist.load_data() #first load the data
#60,000 training examples and 10,000 test, x is training data, y is labels
#X_train & X_test shape is (examples,num_px,num_px) 28x28 greyscale images, Y's shape is (examples,) 
X_train, X_test = X_train/255.0, X_test/255.0 #next, normalize greyscale values between [0,1]

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

model = tf.keras.models.Sequential([ #build up the model layer-by-layer
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10) #leave off softmax activation for now
]) #better to train without softmax to ensure stable loss calculation

#For each example the model returns a vector of "logits" or "log-odds" scores, 
#one for each class.

predictions = model(X_train[:1]) #passing in the first instance from the training set to the untrained model
#yields 10 logits from the last layer after a forward pass. this is saved as a tf.tensor
predictions = model(X_train[:1]).numpy() #instead we can save this output as a numpy array
tf.nn.softmax(predictions).numpy() #pass logits to softmax to get probabilities- convert output to numpy

#a note on slicing
'test1'[:-1] #-1 slice omits the last character '1'
'test1'[:1] #1 slice retains only the first character 't'
#see: https://stackoverflow.com/questions/509211/understanding-slice-notation

#define loss function and specify logits as input with scalar loss as output for each example
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#This loss is equal to the negative log probability of the true class.
#It is zero if the model is sure of the correct class.
#This untrained model gives probabilities close to random (1/10 for each class), so 
#the initial loss should be close to -tf.log(1/10) ~= 2.3. #pass first true val
#and compare with predicted values:
loss_fn(Y_train[:1],predictions).numpy() #returns scalar loss- convert to numpy
#not -2.3, but each time the model is initiated, random weights seem to be applied and the 
#values of this loss function deviate from 1.97 to 2.79 so far. prob avg 2.3.

#prep the model for training
model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])

#model fit adjusts the model parameters to minimize the loss:
model.fit(X_train, Y_train, epochs=5) #train the model using 5 epochs, ends at 0.0732 loss and 0.9770 acc

#model evaluate checks model performance usually on a validation or test set
model.evaluate(X_test, Y_test, verbose=2) #loss is 0.0735, acc is 0.9772

#now if we want the model to return a probability, we can wrap the trained model and attach softmax:
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

probability_model(X_test[:5]) #pass first 5 test instances to new trained model to yield probabilities
#yields a (5,10) tensor with probabilities associated with each node in last layer
