import tensorflow as tf 
import cv2 as cv
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Dropout
import numpy as np


def preprocess(DIR, categories, size, isSave=True ):
    """
    Reads Images in image paths
    Returns
        featureSet -> Image Pixel Values
    Saves the above variables as .npy files
    """
    train = [] 
    try:
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
        train = shuffleTrain(train)

        #Converting to Numpy
        train = np.array(train)

        if isSave == True:
            saveData(train)

    #Returns Training Set
    return train

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

def saveData(x):
    np.save(str(x),x)

def train_val_split(X,y,val_ratio=.2):
    from sklearn.model_selection import train_test_split    
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=val_ratio,random_state = 2)
    return X_train, X_val, y_train, y_val

def normalize(x):
    # Normalizes the data to mean 0 and standard deviation 1
    x = x/255.0
    return x

def createModel(img_size=224, optimizer='adam', batch_size=32, loss='binary_crossentropy'):
    model = Sequential() 
    model.add(Conv2D(32, (3,3), ,activation='relu', input_shape=(img_size,img_size,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, (3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, (3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Converts the 4D output of the Convolutional blocks to a 2D feature which can be read by the Dense layer
    model.add(Flatten())
    # model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))

    # Output Layer
    model.add(Dense(1),activation='sigmoid')

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def imageDataGenerator():
    """
    We are not adding a 'rescale' attribute because the data has already been normalized using the 'normalize' function of this class
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rotation_range=40, 
                                        width_shift_range=.2
                                        height_shift_range=.2
                                        shear_range=.2
                                        zoom_range=.2
                                        horizontal_flip=True
                                        fill_mode='nearest')
    # We do not augment the validation data
    val_datagen = ImageDataGenerator()

    return train_datagen, val_datagen


def saveModel(model, base_name, learn_rate ,attempt):
    model.save_weights(f'{base_name}-{learn_rate}-{attempt}.h5')
    model.save(f'{base_name}_{attempt}.h5')


def plotAcc(histories):
    acc = histories.history['acc']
    val_acc = histories.history['val_acc']
    loss = histories.history['loss']
    val_acc = histories.history['val_loss']

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

def testModel(model):
    X_test, y_test = preprocess(array) # y_test will be empty
    x = np.array(X_test)
    x = normalize(x)
    test_datagen = imageDataGenerator()

    # Plotting
    columns = 5
    i=0
    test_labels = []
    plt.figure(figsize=(30,30))
    for batch in test_datagen.flow(x, batch_size=1):
        pred = model.predict(batch)
        if pred > 0.5:
            test_labels.append(str(categories[1]))
        else:
            test_labels.append(str(categories[0]))
        plt.subplot(5/columns+1, columns, i+1)
        plt.title(f'This is a {test_labels[i]}')
        i += 1
        # Displaying the first 10 images
        if i%10:
            break

        plt.show()