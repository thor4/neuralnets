#clear all variables
%reset -f 

import tensorflow as tf

tf.__version__ #2.4.0

mnist = tf.keras.datasets.mnist #create handle to LeCun's MNIST dataset

(X_train, Y_train), (X_test, Y_test) = mnist.load_data() #first load the data
#60,000 training examples and 10,000 test, x is training data, y is labels
#X_train & X_test shape is (examples,num_px,num_px) 28x28 greyscale images, Y's shape is (examples,) 
X_train, X_test = X_train/255, X_test/255 #next, normalize greyscale values between [0,1]

model = tf.keras.models.Sequential([ #build up the model layer-by-layer
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation='softmax')
]) 

#prep the model for training
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=5) #train the model using 5 epochs, ends at 0.0720 loss and 0.9784 acc

model.evaluate(X_test, Y_test) #loss is 0.0792, acc is 0.9762


## Load+unzip data downloaded from public OSF dataset then remove file and extracted directory


import requests, os
from zipfile import ZipFile

print("Start downloading and unzipping `AnimalFaces` dataset...")
# name = 'AnimalFaces32x32'
# fname = f"{name}.zip"
# nma_url = f"https://osf.io/kgfvj/download"

name = 'test'
fname = f"{name}.zip"
my_url = f"https://osf.io/xt28k/download"
r = requests.get(my_url, allow_redirects=True)
#print(r.headers.get('content-type')) #application/octet-stream for neuromatch

with open(fname, 'wb') as fh:
  fh.write(r.content)

with ZipFile(fname, 'r') as zfile:
  zfile.extractall(f"./{name}")

if os.path.exists(fname):
  os.remove(fname)
else:
  print(f"The file {fname} does not exist")

os.chdir(name)
print("Download completed.")

#this removes the extracted folder:
import shutil
curr_dir = os.getcwd() #if still in extracted folder

try:
    shutil.rmtree(curr_dir)
except OSError as e:
    print("Error: %s : %s" % (dir_path, e.strerror))
