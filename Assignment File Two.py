from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
'''
Develop a spatial convolution model used on image data (mnist) to predict other images in the following way:
    1. set training and test data sets from the mnist data set
    2. reshape data based on the image_data_format() from the input data (both training and test sets)
    3. scale input data for training and test by maximum value in dataset
    4. convert response variables to categories from the number of classes (classes = categories)
    5. develop sequential convolutional model, adding dropout, dense, pooling and flatten layers evaluated for accuracy
    6. fit model to training set, calculating evaluation metrics and loss on test set
    7. fit model to test set and compute overall loss and accuracy
'''

'''
batch_size: number of samples per gradient
num_classes: number of classes used for keras.dense() function
epochs: maximum number of iterations
img_rows: from the mnist data set
img_cols: from the mnist data set
x_train: data from the mnist data set = (60000, 28, 28)
y_train: response data from the mnist data set = (60000,)
x_test: data from the mnist data set = (10000, 28, 28)
y_test: response data from the mnist data set = (10000,)
'''

# set batch size (number of samples per gradient) - used when fitting the model to the training set
batch_size = 128
# set number of classes - used when adding conditions to the model
num_classes = 10
# set epoch (max number of iterations) - used when fitting the model to the training set
epochs = 12

img_rows, img_cols = 28, 28

# create the training and test sets from the mnist dataset __________using how much of a split? 16.7% test___________
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# image_data_format() will return which data format convention keras will follow
if K.image_data_format() == 'channels_first':
    # reshape x_train and x_test for 'channels first' data format
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    # will set input shape for 'channels first' data format - used in the convolution model creation
    input_shape = (1, img_rows, img_cols)
else:
    # reshape x_train and x_test for 'channels last' data format
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    # will set input shape for 'channels last' data format - used in the convolution model creation 
    input_shape = (img_rows, img_cols, 1)

# allows x_train and x_test values to be a floated value (decimals)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# divides input data by 255 (determined from the mnist dataset as max of the datasets)
x_train /= 255
x_test /= 255
# prints shape of x_train
print('x_train shape:', x_train.shape)
# prints number of training samples
print(x_train.shape[0], 'train samples')
# prints number of test samples
print(x_test.shape[0], 'test samples')

# prepares data for compile function to use categorical crossentropy
y_train = keras.utils.to_categorical(y_train, num_classes)
# same as y_train
y_test = keras.utils.to_categorical(y_test, num_classes)

# initializing a sequential model
model = Sequential()
''' add model condition - spatial convolution over images with 32 filters,
     with a 3x3 window, using 'relu' for activation and using the input
     shape specified above '''
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# add second layer movel convolution with 64 layers, same window size, and same activation
model.add(Conv2D(64, (3, 3), activation='relu'))
# add pooling layer - will half each spatial dimension
model.add(MaxPooling2D(pool_size=(2, 2)))
# add dropout layer - will decrease total input units by 1/4 randomly (helps reduce overfitting)
model.add(Dropout(0.25))
''' changes output from model, i.e. model output = (None, 64, 32, 32)
     then flatten() will give model output = (None, 35536) '''
model.add(Flatten())
# add dense layer - will allow for 128-unit input shape under relu activation
model.add(Dense(128, activation='relu'))
# again, will drop half of the input units
model.add(Dropout(0.5))
# again, will allow for number of classes as the input shape under softmax activation
model.add(Dense(num_classes, activation='softmax'))

''' compiles model using categorical crossentropy as the loss function and Adadelta as the optimization function,
     categorical accuracy is used as the metric for evaluating model preformance'''
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

''' fitting the model to the training set, using batch size and epochs set above
     while using the test data as a validation set to calculate loss and accuracy metric'''
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_test, y_test))

# computing the accuracy metrics of the model to the test data
score = model.evaluate(x_test, y_test, verbose=0)
# print test loss from model (total categorical crossentropy value)
print('Test loss:', score[0])
# print test accuracy
print('Test accuracy:', score[1])