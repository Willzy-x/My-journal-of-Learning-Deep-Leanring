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
