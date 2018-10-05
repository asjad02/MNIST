import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

# fix random seed for reproducability
seed = 7
np.random.seed(seed)

# loading data from keras helper function
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# flattening the input image whch is 28*28 dimension into a vector of 28*28=784 elements
num_pixels = X_train.shape[1] * X_train.shape[2]
# X_train(m, height, width)

# we are forcing the precision of the pixel values to be 32bit
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# Feature Scaling enhances the speed
# input is grayscale (0 - 255) 
X_train /= 255
X_test /= 255

# we will be using one hot encoding for this task as we have to get results between 0 to 9
# one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Baseline Model
def baseline_model():
    # model creation
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer="normal", activation='relu'))
    model.add(Dense(num_classes, kernel_initializer="normal", activation="softmax"))

    # Compiling Model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model

# Building the model
model = baseline_model()
# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

#  evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print(scores)
print("Baseline Error: {}".format(100 - scores[1]*100)) 
