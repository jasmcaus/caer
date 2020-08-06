import tensorflow as tf 
import cv2 as cv
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, MaxPooling2D, Conv2D, Dropout
import numpy as np

def preprocess(DIR, categories, size, isSave=False ):
    train = [] 
    featureSet = []
    labels = []
    try:
        featureSet = np.load('featureSet.npy', allow_pickle=True)
        labels = np.load('labels.npy', allow_pickle=True)
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
        train = shuffleTrain(train)

        # Separating the FeatureSet and Labels into separate lists 
        featureSet, labels = sepTrain(train)

        #Converting to Numpy
        train = np.array(train)
        featureSet = np.array(featureSet)
        labels = np.array(labels)

        if isSave == True:
            saveData(featureSet, labels, train)

    #Returns FeatureSet and Labels
    return train, featureSet, labels

def printTotal(count):
    print(count)

def readToGray(image,size):
    try:
        image_array = cv.imread(image)

        # [INFO] Using the following piece of code results in a 'None' in the training set
        # if image_array == None:
        #     pass
        image_gray = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)
        image_gray = cv.resize(image_gray, (size,size))
        return image_gray
    except:
        pass

def shuffleTrain(train):
    import random
    random.shuffle(train)
    return train

def sepTrain(train):
    x = []
    y = []
    for feature, label in train:
        x.append(feature)
        y.append(label)
    return x, y

def saveData(x,y,train):
    np.save('featureSet',x)
    np.save('labels', y)
    np.save('train', train)
    print('[INFO] Numpy Files successfully saved!')

def createModel(img_size=224, optimizer='adam', batch_size=32, loss='binary_crossentropy'):
    model = Sequential() 
    model.add(Conv2D(64, (3,3), input_shape=featureSet.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Converts the 4D output of the Convolutional blocks to a 2D feature which can be read by the Dense layer
    model.add(Flatten())
    model.add(Dense(64))

    # Output Layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model