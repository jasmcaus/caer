# Copyright (c) 2020 Jason Dsouza <jasmcaus@gmail.com>
# Protected under the MIT License (see LICENSE)

#pylint: disable=bare-except

# Importing the necessary packages
import os
import cv2 as cv
import numpy as np
from ._split import train_test_split
from .io_disk import _read_image
from .opencv import to_rgb


def load_img(image_path, target_size=None, channels=1, swapRB=True):
    """
        Loads in an image from `image_path`
    """
    if type(target_size) is not tuple or len(target_size) != 2:
        raise ValueError('[ERROR] target_size must be a tuple of size 2 (width,height')
    
    if type(channels) is not int or channels not in [0,1]:
        raise ValueError('[ERROR] channels must be an integer - 1 (Grayscale) or 3 (RGB)')

    image_array = _read_image(image_path)

    # [INFO] Using the following piece of code results in a 'None' in the training set
    # if image_array == None:
    #     pass
    if channels == 1:
        image_array = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)
    if target_size is not None:
        image_array = cv.resize(image_array, target_size)
    if swapRB:
        image_array = to_rgb(image_array)
        
    return image_array


def get_classes_from_dir(DIR):
    if len(os.listdir(DIR)) == 0:
        raise ValueError('[ERROR] The specified directory does not seem to have any folders in it')
    else:
        classes = [i for i in os.listdir(DIR)]
        return classes
        

def saveNumpy(base_name, data):
    """
    Saves an array to a .npy file
    Converts to Numpy (if not already)
    """

    data = np.array(data)
    if '.npy' in base_name:
        np.save(base_name, data)
    elif '.npz' in base_name:
        np.savez_compressed(base_name, data)


def train_val_split(X, y, val_ratio=.2):
    """
    Do not use if mean subtraction is being employed
    Returns X_train, X_val, y_train, y_val
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio)
    return X_train, X_val, y_train, y_val


def sort_dict(unsorted_dict, descending=False):
    """ 
    Sorts a dictionary in ascending order (if descending = False) or descending order (if descending = True)
    """
    if type(descending) is not bool:
        raise ValueError('[ERROR] `descending` must be a boolean')
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