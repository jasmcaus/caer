# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

# Importing the necessary packages
import sys
sys.path.append('..')

import os
import random
import time
import numpy as np
from .utils import readImg, saveNumpy, get_classes_from_dir
from .preprocessing import MeanProcess, _check_mean_sub_values

def preprocess_from_dir(DIR, 
                        classes=None, 
                        channels=1, 
                        IMG_SIZE=(224,224), 
                        per_class_size=None, 
                        normalize_train=False, 
                        mean_subtraction=None, 
                        isShuffle=True, 
                        save_data=False, 
                        destination_filename=None, 
                        verbose=1):
    """
    Reads Images in base directory DIR using 'classes' (computed from sub directories )
    @param DIR: Directory 
    @param classes --> A list of folder names within `DIR`
    @param channels: Number of channels each image will be processed to (default: 1)
    :param per_class_size: Intended size of the each class to be preprocessed
    :param normalize_train: Whether to normalize each image to between [0,1]
    :param mean_subtraction: Whether mean subtraction should be applied (Tuple)
    :param isShuffle: Shuffle the training set
    :param save_data: If True, saves the training set as a .npy or .npz file based on destination_filename
    :param destination_filename: if save_data is True, the train set will be saved as the filename specified
    :param verbose: Integer either 0 (verbosity off) or 1 (verbosity on). Displays the progress to the terminal as preprocessing continues. Default = 1
    
    :return data: Image Pixel Values with corresponding labels (float32)
    :return classes: ONLY if `classes=None`
    Saves the above variables as .npy files if `save_data = True`
    """
    return_classes_flag = False
    data = [] 

    if not os.path.exists(DIR):
        raise ValueError('[ERROR] The specified directory does not exist')

    if IMG_SIZE is None:
        raise ValueError('[ERROR] IMG_SIZE must be specified')

    if type(IMG_SIZE) is not tuple or len(IMG_SIZE) != 2:
        raise ValueError('[ERROR] IMG_SIZE must be a tuple of size 2 (width,height)')

    if verbose in [0,1]:
        if verbose == 0:
            display_count = False
        else:
            display_count = True
    
    if verbose not in [0,1]:
        raise ValueError('[ERROR] verbose flag must be either 1 (display progress to terminal) or 0 otherwise')

    if type(save_data) is not bool:
        raise ValueError('[ERROR] save_data is a boolean (True/False)')

    if classes is None:
        return_classes_flag = True

    if classes is not None and type(classes) is not list:
        raise ValueError('[ERROR] "classes" must be a list')

    if save_data:
        if destination_filename is None:
            raise ValueError('[ERROR] Specify a destination file name')

        elif not ('.npy' in destination_filename or '.npz' in destination_filename):
            raise ValueError('[ERROR] Specify the correct numpy destination file extension (.npy or .npz)')
    
    if not save_data and destination_filename is not None:
        destination_filename = None
    

    # Loading from Numpy Files
    if destination_filename is not None and os.path.exists(destination_filename):
        since = time.time()
        print('[INFO] Loading from Numpy Files')
        data = np.load(destination_filename, allow_pickle=True)
        end = time.time()
        print('----------------------------------------------')
        print('[INFO] Loaded in {:.0f}s from Numpy Files'.format(end-since))

        return data

    # Extracting image data and adding to `data`
    else:
        since_preprocess = time.time()
        if destination_filename is not None:
            print(f'[INFO] Could not find {destination_filename}. Generating the Image Files')
        else:
            print('[INFO] Could not find a file to load from. Generating Image Files')
        print('----------------------------------------------')

        if classes is None:
            classes = get_classes_from_dir(DIR)

        if per_class_size is None:
            per_class_size = len(os.listdir(os.path.join(DIR, classes[0])))

        if mean_subtraction is not None:
            # Checking if 'mean_subtraction' values are valid. Returns boolean value
            subtract_mean = _check_mean_sub_values(mean_subtraction, channels)

        for item in classes:
            class_path = os.path.join(DIR, item)
            class_label = classes.index(item)
            count = 0 
            for image in os.listdir(class_path):
                if count != per_class_size:
                    image_path = os.path.join(class_path, image)

                    # Returns the resized image
                    img = readImg(image_path, resized_img_size=IMG_SIZE, channels=channels)
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
            
            since = time.time()
            # Saving
            saveNumpy(destination_filename, data)
            end = time.time()
            
            time_elapsed = end-since

            print('[INFO] {} saved! Took {:.0f}m {:.0f}s'.format(destination_filename, time_elapsed // 60, time_elapsed % 60))

        #Returns Training Set
        end_preprocess = time.time()
        time_elapsed_preprocess = end_preprocess-since_preprocess
        print('----------------------------------------------')
        print('[INFO] {} files preprocessed! Took {:.0f}m {:.0f}s'.format(count, time_elapsed_preprocess // 60, time_elapsed_preprocess % 60))

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
    
    if IMG_SIZE is None:
        raise ValueError('[ERROR] IMG_SIZE not defined')

    if type(IMG_SIZE) is not tuple or len(IMG_SIZE) != 2:
        raise ValueError('[ERROR] IMG_SIZE must be a tuple of size 2')

    else:
        x = [i[0] for i in data]
        y = [i[1] for i in data]

        # Without reshaping, X.shape --> (no. of images, IMG_SIZE, IMG_SIZE)
        # On reshaping, X.shape --> (no. of images, IMG_SIZE, IMG_SIZE,channels)

        # Converting to Numpy + Reshaping X
        x = reshape(x, IMG_SIZE, channels)
        y = np.array(y)

        return x, y


def reshape(x, IMG_SIZE, channels):
    if IMG_SIZE is None:
        raise ValueError('[ERROR] IMG_SIZE not defined')

    if type(IMG_SIZE) is not tuple or len(IMG_SIZE) != 2:
        raise ValueError('[ERROR] IMG_SIZE must be a tuple of size 2')

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
