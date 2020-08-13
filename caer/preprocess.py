# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

# Importing the necessary packages
import numpy as np
from .convenience import readToGray
from .convenience import saveNumpy


def preprocess(DIR, categories, size, isSave=True ):
    """
    Reads Images in image paths
    Returns
        train -> Image Pixel Values with corresponding labels
    Saves the above variables as .npy files
    """

    train = [] 
    try:
        # If train.npy already exists, load it in
        # if os.path.exists('featureSet.npy')
        train = np.load('train.npy', allow_pickle=True)
        print('[INFO] Loading from Numpy Files')
    except:
        print('[INFO] Generating the Image Files')
        for category in categories:
            category_path = os.path.join(DIR, category)
            classNum = categories.index(category)
            count = 0 
            for image in os.listdir(category_path):
                if count != size:
                    image_path = os.path.join(category_path, image)
                    gray = readToGray(image_path, 100)

                    train.append([gray, classNum])
                    count +=1 
                    printTotal(count)
                else:
                    break
        # Shuffling the Training Set
        train = shuffle(train)

        #Converting to Numpy
        train = np.array(train)

        # Saves the Train set as a .npy file
        if isSave == True:
            saveNumpy(train)

    #Returns Training Set
    return train

def printTotal(count):
    print(count)

def shuffle(train):
    import random
    random.shuffle(train)
    return train

def sepTrain(train,IMG_SIZE=224,channels=1):
    # x = []
    # y = []
    # for feature, label in train:
    #     x.append(feature)
    #     y.append(label)

    x = [i[0] for i in train]
    y = [i[1] for i in train]

    # Converting to Numpy
    x = np.array(x).reshape(-1,IMG_SIZE,IMG_SIZE,channels)
    y = np.array(y)

    return x, y

def normalize(x):
    """
    Normalizes the data to mean 0 and standard deviation 1
    """
    x = x/255.0
    return x

def imageDataGenerator():
    """
    We are not adding a 'rescale' attribute because the data has already been normalized using the 'normalize' function of this class

    Returns train_datagen, val_datagen
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rotation_range=40, 
                                        width_shift_range=.2,
                                        height_shift_range=.2,
                                        shear_range=.2,
                                        zoom_range=.2,
                                        horizontal_flip=True,
                                        fill_mode='nearest')
    # We do not augment the validation data
    val_datagen = ImageDataGenerator()

    return train_datagen, val_datagen