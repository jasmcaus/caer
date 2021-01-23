#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-21 The Caer Authors <http://github.com/jasmcaus>


import random
import time
import numpy as np

from .core import get_classes_from_dir, to_array
from .io import imread
from .preprocessing import MeanProcess
from ._internal import _check_target_size, _check_mean_sub_values
from .path import listdir, minijoin, exists, list_images
from .color import to_gray

__all__ = [
    'preprocess_from_dir',
    'sep_train',
    'shuffle',
    'reshape',
    'normalize'
]


def preprocess_from_dir(DIR, 
                        classes=None, 
                        IMG_SIZE=(224,224), 
                        channels=3, 
                        per_class_size=None, 
                        normalize_train=False, 
                        mean_subtraction=None, 
                        isShuffle=True, 
                        save_data=False, 
                        destination_filename=None, 
                        verbose=1):
    """
    Reads Images in base directory DIR using 'classes' (computed from sub directories )
    Arguments:
        :param DIR: Base directory 
        :param classes: A list of folder names within `DIR`. Automatically inferred from DIR if not provided
        :param IMG_SIZE: Image Size tuple of size 2 (width, height)
        :param channels: Number of channels each image will be processed to (default: 3)
        :param per_class_size: Intended size of the each class to be preprocessed
        :param normalize_train: Whether to normalize each image to between [0,1]
        :param mean_subtraction: Whether mean subtraction should be applied (Tuple)
        :param isShuffle: Shuffle the training set
        :param save_data: If True, saves the training set as a .npy or .npz file based on destination_filename
        :param destination_filename: if save_data is True, the train set will be saved as the filename specified
        :param verbose: Integer either 0 (verbosity off) or 1 (verbosity on). Displays the progress to the terminal as preprocessing continues. Default = 1
    
    Returns
        :return data: Image Pixel Values with corresponding labels (float32)
        :return classes: ONLY if `classes=None`
        Saves the above variables as .npy files if `save_data = True`
    """
    return_classes_flag = False
    data = [] 

    if not exists(DIR):
        raise ValueError('The specified directory does not exist')

    if IMG_SIZE is None:
        raise ValueError('IMG_SIZE must be specified')

    if not isinstance(IMG_SIZE, tuple) or len(IMG_SIZE) != 2:
        raise ValueError('IMG_SIZE must be a tuple of size 2 (width,height)')

    if verbose in [0,1]:
        if verbose == 0:
            display_count = False
        else:
            display_count = True
    
    else:
        raise ValueError('verbose flag must be either 1 (display progress to terminal) or 0 otherwise')

    if not isinstance(save_data, bool):
        raise ValueError('save_data must be a boolean (True/False)')

    if classes is None:
        return_classes_flag = True

    else:
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
    if destination_filename is not None and exists(destination_filename):
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

        if classes is None:
            classes = get_classes_from_dir(DIR)            
            # Removing false folders
            classes = _check_for_false_folders(DIR, classes)

        if per_class_size is None:
            per_class_size = len(listdir(minijoin(DIR, classes[0]), verbose=0))

        if mean_subtraction is not None:
            # Checking if 'mean_subtraction' values are valid. Returns boolean value
            subtract_mean = _check_mean_sub_values(mean_subtraction, channels)

        for item in classes:
            class_path = minijoin(DIR, item)
            class_label = classes.index(item)
            count = 0 
            tens_list = list_images(class_path, use_fullpath=True, verbose=0)

            for image_path in tens_list:
                if count != per_class_size:
                    # image_path = minijoin(class_path, image)

                    # Returns the resized image (ignoring aspect ratio since it isn't relevant for Deep Computer Vision models)
                    tens = imread(image_path, target_size=IMG_SIZE, rgb=True)

                    if tens is None:
                        continue
                    
                    # Gray
                    if channels == 1:
                        tens = to_gray(tens)

                    # Normalizing
                    if normalize_train:
                        tens = normalize(tens)
                    
                    # Subtracting Mean
                    # Mean must be calculated ONLY on the training set
                    if mean_subtraction is not None and subtract_mean:
                        mean_subtract = MeanProcess(mean_subtraction, channels)
                        tens = mean_subtract.mean_preprocess(tens, channels)
                        
                    # Appending to train set
                    data.append([tens, class_label])
                    count +=1 

                    if display_count is True:
                        _printTotal(count, item)
                else:
                    break

        # Shuffling the Training Set
        if isShuffle is True:
            data = shuffle(data)

        # Converting to Numpy
        data = to_array(data)

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

        if return_classes_flag:
            return data, classes
        else:
            return data


def _printTotal(count, category):
    print(f'{count} - {category}')


def _check_for_false_folders(base_dir, classes):
    if len(classes) == 0:
        return []
    else:
        g = []
        for i in classes:
            path = minijoin(base_dir, i)
            if list_images(path, recursive=False, verbose=0) is not None:
                g.append(i)
    
    return g


def shuffle(data):
    """
    Shuffles the Array
    """
    random.shuffle(data)
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
    y = to_array(y)

    return x, y


def reshape(x, IMG_SIZE, channels):
    _ = _check_target_size(IMG_SIZE)

    width, height = IMG_SIZE[:2]
    return to_array(x).reshape(-1, width, height, channels)


def normalize(x, dtype='float32'):
    """
    Normalizes the data to mean 0 and standard deviation 1
    """
    # x/=255.0 raises a TypeError
    # x = x/255.0
    
    # Converting to float32 and normalizing (float32 saves memory)
    x = x.astype(dtype) / 255
    return x