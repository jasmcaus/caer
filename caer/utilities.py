#
#  _____ _____ _____ _____
# |     |     | ___  | __|  Caer - Modern Computer Vision
# |     | ___ |      | \    Languages: Python, C, C++
# |_____|     | ____ |  \   http://github.com/jasmcaus/caer

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>


#pylint: disable=bare-except

import numpy as np

from ._split import train_test_split
from .path import listdir, minijoin


def median(arr, axis=None):
    return np.median(arr, axis=axis)


def npmean(arr):
    return np.mean(arr)


def array(obj, dtype=None, order='K'):
    return np.array(obj, dtype=dtype, order=order)


def to_array(obj, dtype=None, order='K'):
    return np.array(obj, dtype=dtype, order=order)


def asarray(obj, dtype=None, order=None):
    return np.asarray(obj, dtype=dtype, order=order)


def load(filename, allow_pickle=False):
    return np.load(filename, allow_pickle=allow_pickle)


def get_classes_from_dir(DIR, verbose=0):
    if len(listdir(DIR, verbose=0)) == 0:
        raise ValueError('The specified directory does not seem to have any folders in it')
    else:
        import os 
        classes = [i for i in listdir(DIR, include_subdirs=False, verbose=verbose) if os.path.isdir(minijoin(DIR, i))]
        return classes
        

def saveNumpy(base_name, data):
    """
    Saves an array to a .npy file
    Converts to Numpy (if not already)
    """
    if not (isinstance(data, list) or isinstance(data, np.ndarray)):
        raise ValueError('data needs to be a Python list or a Numpy array')

    data = np.array(data)
    if '.npy' in base_name:
        np.save(base_name, data)
        print(f'[INFO] {base_name} saved!')
    elif '.npz' in base_name:
        np.savez_compressed(base_name, data)
        print(f'[INFO] {base_name} saved!')


def train_val_split(X, y, val_ratio=.2):
    """
    Do not use if mean subtraction is being employed
    Returns X_train, X_val, y_train, y_val
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio)
    return X_train, y_train, X_val, y_val


def sort_dict(unsorted_dict, descending=False):
    """ 
    Sorts a dictionary in ascending order (if descending = False) or descending order (if descending = True)
    """
    if isinstance(descending, bool):
        raise ValueError('`descending` must be a boolean')
    return sorted(unsorted_dict.items(), key=lambda x:x[1], reverse=descending)


# def plotAcc(histories):
#     """
#     Plots the model accuracies as 2 graphs
#     """
#     pass
    # import matplotlib.pyplot as plt 
    # acc = histories.history['acc']
    # val_acc = histories.history['val_acc']
    # loss = histories.history['loss']
    # val_loss = histories.history['val_loss']

    # epochs = range(1, len(acc)+1)

    # # Plotting Accuracy
    # plt.plot(epochs, acc, 'b', label='Training Accuracy')
    # plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    # plt.title('Training and Validation Accuracy')
    # plt.legend()

    # # Plotting Loss
    # plt.plot(epochs, loss, 'b', label='Training Loss')
    # plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    # plt.title('Training and Validation Loss')
    # plt.legend()

    # plt.show()


__all__ = [
    'get_classes_from_dir',
    'median',
    'npmean',
    'asarray',
    'to_array',
    'array',
    'saveNumpy',
    'train_val_split',
    'sort_dict'
    # 'plotAcc'
]