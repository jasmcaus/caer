# Importing the necessary packages
import cv2 as cv
import os
import numpy as np

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

def saveNumpy(x):
    """
    Saves an array to a .npy file
    Converts to Numpy (if not already)
    """
    import numpy as np
    x = np.array(x)
    np.save(str(x),x)

def train_val_split(X,y,val_ratio=.2):
    """
    Returns X_train, X_val, y_train, y_val
    """
    from sklearn.model_selection import train_test_split    
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=val_ratio,random_state = 2)
    return X_train, X_val, y_train, y_val

def plotAcc(histories):
    """
    Plots the model accuracies as 2 graphs
    """
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

