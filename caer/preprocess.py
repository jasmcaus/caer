# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

# Importing the necessary packages
import os
import time
import numpy as np
from .utils import readToGray
from .utils import saveNumpy

def preprocess(DIR, classes, name, IMG_SIZE=224, train_size=None, isNormalize=False, isShuffle=True, isSave = True):
    """
    Reads Images in base directory DIR
    Returns
        train -> Image Pixel Values with corresponding labels
    Saves the above variables as .npy files if isSave = True
    """

    train = [] 
    try:
        if isSave is True and not ('.npy' in name or '.npz' in name):
            print('[ERROR] Specify the correct numpy destination file extension (.npy or .npz)')
            raise TypeError
            
        elif os.path.exists(name):
            since = time.time()
            print('[INFO] Loading from Numpy Files')
            train = np.load(name, allow_pickle=True)
            end = time.time()
            print('[INFO] Loaded in {:.0f}s from Numpy Files'.format(end-since))

            # Raising TypeError although not a TypeError to escape from the try block
            # Alternatively, use the 'pass' keyword
            # raise TypeError

            return train

        else:
            since_preprocess = time.time()
            print(f'[INFO] Could not find {name}. Generating the Image Files')

            if train_size is None:
                train_size = len(os.listdir(os.path.join(DIR, classes[0])))

            for item in classes:
                class_path = os.path.join(DIR, item)
                classNum = classes.index(item)
                count = 0 
                for image in os.listdir(class_path):
                    if count != train_size:
                        image_path = os.path.join(class_path, image)

                        # Returns image RESIZED and GRAY
                        gray = readToGray(image_path, IMG_SIZE)
                        # Normalizing
                        if isNormalize is True:
                            gray = normalize(gray)
                            
                        train.append([gray, classNum])
                        count +=1 
                        _printTotal(count, item)
                    else:
                        break

            # Shuffling the Training Set
            if isShuffle is True:
                train = shuffle(train)

            # Converting to Numpy
            train = np.array(train)

            # Saves the Train set as a .npy file
            if isSave is True:
                #Converts to Numpy and saves
                if '.npy' in name:
                    print('[INFO] Saving as .npy file')
                elif '.npz' in name:
                    print('[INFO] Saving as .npz file')
                since = time.time()
                
                # Saving
                saveNumpy(name, train)

                end = time.time()
                time_elapsed = end-since
                if '.npy' in name:
                    print('{}.npy saved! Took {:.0f}m {:.0f}s'.format(name, time_elapsed // 60, time_elapsed % 60))

                elif '.npz' in name:
                    print('{}.npz saved! Took {:.0f}m {:.0f}s'.format(name, time_elapsed // 60, time_elapsed % 60))

            #Returns Training Set
            end_preprocess = time.time()
            time_elapsed_preprocess = end_preprocess-since_preprocess
            print('Preprocessing complete! Took {:.0f}m {:.0f}s'.format(time_elapsed_preprocess // 60, time_elapsed_preprocess % 60))

            return train

    except TypeError:
        pass


def _printTotal(count, category):
    print(f'{count} - {category}')

def shuffle(train):
    """
    Shuffles the Array
    """
    import random
    random.shuffle(train)
    return train

def sepTrain(train, IMG_SIZE=224, channels=1):
    # x = []
    # y = []
    # for feature, label in train:
    #     x.append(feature)
    #     y.append(label)

    x = [i[0] for i in train]
    y = [i[1] for i in train]

    # Without reshaping, X.shape --> (no. of images, IMG_SIZE, IMG_SIZE)
    # On reshaping, X.shape --> (no. of images, IMG_SIZE, IMG_SIZE,channels)

    # Converting to Numpy + Reshaping X
    x = reshape(x, IMG_SIZE, channels)
    y = np.array(y)

    return x, y

def reshape(x, IMG_SIZE, channels):
    return np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, channels)

def normalize(x):
    """
    Normalizes the data to mean 0 and standard deviation 1
    """
    x = x/255.0
    return x