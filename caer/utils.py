# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

#pylint: disable=bare-except

# Importing the necessary packages
import os
import cv2 as cv
import numpy as np
from ._split import train_test_split

# # For Python 2.7
# import sys
# if sys.version_info.major == 2:
#     from urllib2 import urlopen
# For Python

def readImg(image_path, resized_img_size=None, channels=1):
    if not os.path.exists(image_path):
        raise FileNotFoundError('[ERROR] The image file was not found')

    image_array = cv.imread(image_path)

    # [INFO] Using the following piece of code results in a 'None' in the training set
    # if image_array == None:
    #     pass
    if channels == 1:
        image_array = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)
    if resized_img_size is not None:
        image_array = cv.resize(image_array, (resized_img_size,resized_img_size))
    return image_array

def get_classes_from_dir(DIR):
    if len(os.listdir(DIR)) == 0:
        raise ValueError('[ERROR] The specified directory does not seem to have any folders in it')
    else:
        classes = [i for i in os.listdir(DIR)]
        return classes
        

def saveNumpy(name, x):
    """
    Saves an array to a .npy file
    Converts to Numpy (if not already)
    """

    x = np.array(x)
    if '.npy' in name:
        np.save(name, x)
    elif '.npz' in name:
        np.savez_compressed(name, x)


def train_val_split(X,y,val_ratio=.2):
    """
    Do not use if mean subtraction is being employed
    Returns X_train, X_val, y_train, y_val
    """
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=val_ratio)
    return X_train, X_val, y_train, y_val


def sort_dict(unsorted_dict, descending=False):
    """ 
    Sorts a dictionary in ascending order (if descending = False) or descending order (if descending = True)
    """
    return sorted(unsorted_dict.items(), key=lambda x:x[1], reverse=descending)

def plotAcc(histories):
    """
    Plots the model accuracies as 2 graphs
    """
    import matplotlib.pyplot as plt 
    acc = histories.history['acc']
    val_acc = histories.history['val_acc']
    loss = histories.history['loss']
    val_loss = histories.history['val_loss']

    epochs = range(1, len(acc)+1)

    # Plotting Accuracy
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plotting Loss
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()