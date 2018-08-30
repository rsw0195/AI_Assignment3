import numpy as np 
from sklearn.datasets import load_iris #unused 
from sknn.mlp import Classifier, Layer 
from sklearn import preprocessing 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import model_selection #unused
from sklearn import datasets
import warnings
'''
Create a predictive classification model for the IRIS dataset using an 80/20 split. The conclusion will be printed values 
for the accuracy of the training model, and the accuracy of the test model. General layout of this model will include:
    1. Splitting the dataset using the 80/20 split and the train_test_split() function from the SKLearn package
    2. Scaling the input values for both the training and test set using the l2-norm and then adjusting those values around the mean
    3. Developing the classification model using rectifier as the nonlinear input activation method,
        stochastic gradient descent as the learning method, and softmax as the linear output activation method
    4. Fit the model around the training set
    5. Predict the response variables for the test set
    6. Using cross validation with K=5, determine the accuracy of the training model.
    7. Print the training model accuracy (take average of the 5 accuracy values from above)
    8. Calculate and print the test model accuracy by comparing the predicted values from step 5 and the actual values from the test set
'''

'''
iris: dataset used from the SKLearn package predicting the name of the flower based on various flower descriptors
x_train: numpy array of descriptor values for 80% of the dataset
y_train: numpy array of the target values for those descriptors in the x_train set
x_test: numpy array of the remaining data points used to evaluate success of the model
y_test: numpy array of the remaining target data points used to evaluate the success of the model
'''


# ignores any deprication warnings from packages above that may have updates
warnings.simplefilter("ignore", category=DeprecationWarning)

# import that iris dataset under the SKLearn package    
iris = datasets.load_iris()
# split the iris dataset into training and test sets for an 80/20 split. 
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)

# scale the x_train dataset under the l2-norm
X_trainn = preprocessing.normalize(X_train, norm='l2')
# scale the x_test dataset under the l2-norm
X_testn = preprocessing.normalize(X_test, norm='l2')

# scale the x_trainn dataset around the mean
X_trainn = preprocessing.scale(X_trainn)
#scale the x_testn dataset around the mean
X_testn = preprocessing.scale(X_testn)

# use a classification neural network to create predictive model
clsfr = Classifier(
            layers=[
            # Rectifier is used for both nonlinear input activation layers using 13 units
            Layer("Rectifier", units=13),   
            Layer("Rectifier", units=13),   
            ''' Softmax is used as the linear output activation layer - form of linear regression
                 using mutually exclusive multi-class classification responses'''
            Layer("Softmax")],
            # learning rate parameter set at 0.001
            learning_rate=0.001,
            # learning rule using the stochastic gradient descent to minimize the objective function
            learning_rule='sgd',
            # random seed set for classification model
            random_state=201,
            # max number of iterations used to develop model (n_iter = epoch)
            n_iter=200)

# predictive model fit around the training set, evaluated over both the scaled x and unscaled y
model1=clsfr.fit(X_trainn, y_train)
# Using the predictive model above to predict the target from the scaled (x) test set
y_hat=clsfr.predict(X_testn)
# computes the accuracy of the training model as compared to the training target data using K-fold CV for 5-folds 
scores = cross_val_score(clsfr, X_trainn, y_train, cv=5)
# print the model accuracy scores for each iteration (5)
print scores
# prints the average (mean) accuracy of the training model from the cross_val_score() function output
print 'train mean accuracy %s' % np.mean(scores)
# prints the test accuracy between the predicted values and the actual values
print 'vanilla sgd test %s' % accuracy_score(y_hat,y_test)