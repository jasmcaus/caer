#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++, Cuda
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>


import os
import random
import time
import numpy as np

from .io import imread
from ._internal import _check_target_size
from .path import minijoin, list_images
from .color import to_gray

__all__ = [
    'preprocess_from_dir',
    'sep_train',
]


def preprocess_from_dir(DIR, 
                        classes=None, 
                        IMG_SIZE=None, 
                        channels=3, 
                        isShuffle=True, 
                        save_data=False, 
                        destination_filename=None, 
                        verbose=True):
    """
    Reads Images in base directory DIR using 'classes' (computed from sub directories )
    Arguments:
        DIR (str): Base directory 
        classes (list): A list of folder names within `DIR`.
        IMG_SIZE (tuple): Image Size tuple of size 2 (width, height)
        channels (int): Number of channels each image will be processed to (default: 3)
        isShuffle (bool): Shuffle the training set
        save_data (bool): If True, saves the training set as a .npy or .npz file based on destination_filename
        destination_filename (str): if save_data is True, the train set will be saved as the filename specified
        verbose (bool): Displays the progress to the terminal as preprocessing continues. Default = True
    
    Returns
        data: Image Pixel Values with corresponding labels (float32)
        Saves the above variables as .npy files if `save_data = True`
    """
    data = [] 

    if not os.path.exists(DIR):
        raise ValueError('The specified directory does not exist')

    if IMG_SIZE is None:
        raise ValueError('IMG_SIZE must be specified')

    if not isinstance(IMG_SIZE, tuple) or len(IMG_SIZE) != 2:
        raise ValueError('IMG_SIZE must be a tuple of size 2 (width,height)')
    else:
        raise ValueError('verbose flag must be either 1 (display progress to terminal) or 0 otherwise')

    if not isinstance(save_data, bool):
        raise ValueError('save_data must be a boolean (True/False)')

    if not isinstance(classes, list):
        raise ValueError('"classes" must be a list')

    if save_data:
        if destination_filename is None:
            raise ValueError('Specify a destination file name')

        elif not ('.npy' in destination_filename or '.npz' in destination_filename):
            raise ValueError('Specify the correct numpy destination file extension (.npy or .npz)')
    
    if not save_data and destination_filename is not None:
        destination_filename = None

    # Loading from Numpy Files
    if destination_filename is not None and os.path.exists(destination_filename):
        print('[INFO] Loading from Numpy Files')
        since = time.time()
        data = np.load(destination_filename, allow_pickle=True)
        end = time.time()
        took = end - since
        print('----------------------------------------------')
        print(f'[INFO] Loaded in {took:.0f}s from Numpy Files')

        return data

    # Extracting image data and adding to `data`
    else:
        if destination_filename is not None:
            print(f'[INFO] Could not find {destination_filename}. Generating the training data')
        else:
            print('[INFO] Could not find a file to load from. Generating the training data')
        print('----------------------------------------------')

        # Starting timer
        since_preprocess = time.time()

        for item in classes:
            class_path = minijoin(DIR, item)
            class_label = classes.index(item)
            count = 0 
            tens_list = list_images(class_path, use_fullpath=True, verbose=0)

            for image_path in tens_list:
                tens = imread(image_path, target_size=IMG_SIZE, rgb=True)

                if tens is None:
                    continue
                
                # Gray
                if channels == 1:
                    tens = to_gray(tens)

                # Appending to train set
                data.append([tens, class_label])
                count += 1

        # Shuffling the Training Set
        if isShuffle is True:
            random.shuffle(data)

        # Converting to Numpy
        data = np.array(data)

        # Saves the Data set as a .npy file
        if save_data:
            #Converts to Numpy and saves
            if destination_filename.endswith('.npy'):
                print('[INFO] Saving as .npy file')
            elif destination_filename.endswith('.npz'):
                print('[INFO] Saving as .npz file')
            
            # Saving
            since = time.time()
            np.save(destination_filename, data)
            end = time.time()
            
            time_elapsed = end-since
            minu_elapsed = time_elapsed // 60
            sec_elapsed = time_elapsed % 60
            print(f'[INFO] {destination_filename} saved! Took {minu_elapsed:.0f}m {sec_elapsed:.0f}s')

        #Returns Training Set
        end_preprocess = time.time()
        time_elapsed_preprocess = end_preprocess - since_preprocess
        minu = time_elapsed_preprocess // 60
        sec = time_elapsed_preprocess % 60

        print('----------------------------------------------')
        print(f'[INFO] {len(data)} files preprocessed! Took {minu:.0f}m {sec:.0f}s')

        return data


def sep_train(data, IMG_SIZE, channels=1):
    # x = []
    # y = []
    # for feature, label in data:
    #     x.append(feature)
    #     y.append(label)
    
    _ = _check_target_size(IMG_SIZE)

    x = [i[0] for i in data]
    y = [i[1] for i in data]

    # Without reshaping, X.shape --> (no. of images, IMG_SIZE, IMG_SIZE)
    # On reshaping, X.shape --> (no. of images, IMG_SIZE, IMG_SIZE, channels)

    # Converting to Numpy + Reshaping X
    x = reshape(x, IMG_SIZE, channels)
    y = np.array(y)

    return x, y


def reshape(x, IMG_SIZE, channels):
    _ = _check_target_size(IMG_SIZE)

    width, height = IMG_SIZE[:2]
    return np.array(x).reshape(-1, width, height, channels)