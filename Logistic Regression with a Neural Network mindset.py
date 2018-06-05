import numpy as np
import matplotlib.pyplot as plt

import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

%matplotlib inline

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

#Example of a picture
#We added "_orig" at the end of image datasets (train and test) because we are going to preprocess them.
#After preprocessing, we will end up with train_set_x and test_set_x (the labels train_set_y and test_set_y don't need any preprocessing).
index = 15
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

#Many software bugs in deep learning come from having matrix/vector dimensions that don't fit. 
#If you can keep your matrix/vector dimensions straight you will go a long way toward eliminating many bugs.
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1
                                
#For convenience, you should now reshape images of shape (num_px, num_px, 3) in a numpy-array of shape (num_px  ∗∗  num_px  ∗∗  3, 1).
#After this, our training (and test) dataset is a numpy-array where each column represents a flattened image. 
#There should be m_train (respectively m_test) columns.
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

#The main steps for building a Neural Network are:

#Define the model structure (such as number of input features)
#Initialize the model's parameters Loop:
#   Calculate current loss (forward propagation)
#   Calculate current gradient (backward propagation)
#   Update parameters (gradient descent)
                                
def sigmoid(z):
    s = 1/(1+(-1*np.exp(z)))
    return s

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))                            
    b = 0
    assert(w.shape == (dim,1))
    assert(isinstance(b,float), or isinstance(b,int))
                                
    return w,b
       
#Implement a function propagate() that computes the cost function and its gradient.                                
def propagate(w, b, X, Y):
    m = X.shape[1]
    
    A = sigmoid(np.dot(w.T, X)+b)
    cost = (-1/m)*(np.dot(Y, np.log(A))+np.dot(1-Y, np.log(1-A)))            
    dw = (1/m)*(np.dot(X, (A-Y).T))
    db = (1/m)*(np.sum(A-Y))   
                                
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost 
                                
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    
    for i in range(num_iterations):         
        grads,cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate*dw
        b = b - learning_rate*db
                                
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs     
                                
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)                            
                                
    A= sigmoid(np.dot(w.T, X) + b)
    
    for i in range(A.shape[1]):
        if(A[0, i] > 0.5):
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0         
                                
    assert(Y_prediction.shape == (1, m))
    return Y_prediction
                                
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):                                
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """          
    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost = False)
    w = parameters["w"]
    b = parameters["b"]                            
                                
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)                            
                                
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))                            
                                
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
                                
#Choice of learning rate
#Reminder:
#In order for Gradient Descent to work you must choose the learning rate wisely. 
#The learning rate  αα  determines how rapidly we update the parameters. 
#If the learning rate is too large we may "overshoot" the optimal value. Similarly, 
#if it is too small we will need too many iterations to converge to the best values. 
#That's why it is crucial to use a well-tuned learning rate.                                
