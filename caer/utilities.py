#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>


#pylint: disable=bare-except

import numpy as np

from .path import listdir, minijoin

__all__ = [
    'get_classes_from_dir',
    'median',
    'npmean',
    'asarray',
    'to_array',
    'array',
    'train_val_split',
    'sort_dict'
]


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


def get_classes_from_dir(DIR, verbose=0):
    if len(listdir(DIR, verbose=0)) == 0:
        raise ValueError('The specified directory does not seem to have any folders in it')
    else:
        import os 
        classes = [i for i in listdir(DIR, include_subdirs=False, verbose=verbose) if os.path.isdir(minijoin(DIR, i))]
        return classes


def _sep(data):
    
    x = [i[0] for i in data]
    y = [i[1] for i in data]

    return x, y


def train_val_split(X, y, val_ratio=.2):
    """
        Do not use if mean subtraction is being employed
        Returns X_train, X_val, y_train, y_val
    """
    if len(X) != len(y):
        raise ValueError('X must be equal to y')
    
    data = [] 
    for i in range(len(X)):
        data.append([X[i], y[i]])
    
    # import random
    # random.shuffle(data)

    split = int(len(X) - (len(X) * val_ratio)) - 1

    data_train = data[0:split]
    data_test = data[split:]

    X_train, y_train = _sep(data_train)
    X_val, y_val = _sep(data_test)

    return X_train, X_val, y_train, y_val


def sort_dict(unsorted_dict, descending=False):
    """ 
        Sorts a dictionary in ascending order (if descending = False) or descending order (if descending = True)
    """
    if not isinstance(descending, bool):
        raise ValueError('`descending` must be a boolean')

    return sorted(unsorted_dict.items(), key=lambda x:x[1], reverse=descending)