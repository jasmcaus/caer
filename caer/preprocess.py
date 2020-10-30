# Copyright 2020 The Caer Authors. All Rights Reserved.
#
# Licensed under the MIT License (see LICENSE);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at <https://opensource.org/licenses/MIT>
#
# ==============================================================================

import random
import time
import numpy as np

from .utilities import saveNumpy, get_classes_from_dir
from .images import load_img
from .preprocessing import MeanProcess
from ._checks import _check_target_size, _check_mean_sub_values
from .path import listdir, minijoin, exists, list_images


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
        print('----------------------------------------------')
        print('[INFO] Loaded in {:.0f}s from Numpy Files'.format(end-since))

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

        if per_class_size is None:
            per_class_size = len(listdir(minijoin(DIR, classes[0])))

        if mean_subtraction is not None:
            # Checking if 'mean_subtraction' values are valid. Returns boolean value
            subtract_mean = _check_mean_sub_values(mean_subtraction, channels)

        for item in classes:
            class_path = minijoin(DIR, item)
            class_label = classes.index(item)
            count = 0 

            for image_path in list_images(class_path, use_fullpath=True, verbose=0):
                if count != per_class_size:
                    # image_path = minijoin(class_path, image)

                    # Returns the resized image (ignoring aspect ratio since it isn't relevant for Deep Computer Vision models)
                    img = load_img(image_path, target_size=IMG_SIZE, channels=channels)
                    if img is None:
                        continue

                    # Normalizing
                    if normalize_train:
                        img = normalize(img)
                    
                    # Subtracting Mean
                    # Mean must be calculated ONLY on the training set
                    if mean_subtraction is not None and subtract_mean:
                        mean_subtract = MeanProcess(mean_subtraction, channels)
                        img = mean_subtract.mean_preprocess(img, channels)
                        
                    # Appending to train set
                    data.append([img, class_label])
                    count +=1 

                    if display_count is True:
                        _printTotal(count, item)
                else:
                    break

        # Shuffling the Training Set
        if isShuffle is True:
            data = shuffle(data)

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
            saveNumpy(destination_filename, data)
            end = time.time()
            
            time_elapsed = end-since

            print('[INFO] {} saved! Took {:.0f}m {:.0f}s'.format(destination_filename, time_elapsed // 60, time_elapsed % 60))

        #Returns Training Set
        end_preprocess = time.time()
        time_elapsed_preprocess = end_preprocess - since_preprocess
        print('----------------------------------------------')
        print('[INFO] {} files preprocessed! Took {:.0f}m {:.0f}s'.format(len(data), time_elapsed_preprocess // 60, time_elapsed_preprocess % 60))

        if return_classes_flag:
            return data, classes
        else:
            return data


def _printTotal(count, category):
    print(f'{count} - {category}')


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
    y = np.array(y)

    return x, y


def reshape(x, IMG_SIZE, channels):
    _ = _check_target_size(IMG_SIZE)

    width, height = IMG_SIZE[:2]
    return np.array(x).reshape(-1, width, height, channels)


def normalize(x, dtype='float32'):
    """
    Normalizes the data to mean 0 and standard deviation 1
    """
    # x/=255.0 raises a TypeError
    # x = x/255.0
    
    # Converting to float32 and normalizing (float32 saves memory)
    x = x.astype(dtype) / 255
    return x


__all__ = [
    'preprocess_from_dir',
    'sep_train',
    'shuffle',
    'reshape',
    'normalize'
]