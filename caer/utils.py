# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

#pylint: disable=bare-except

# Importing the necessary packages
# import os
from urllib.request import urlopen
import os
import cv2 as cv
import numpy as np

# # For Python 2.7
# import sys
# if sys.version_info.major == 2:
#     from urllib2 import urlopen
# For Python

def readImg(image_path, resized_img_size=None, channels=1):
    if not os.path.exists(image_path):
        raise FileNotFoundError('[ERROR] The image file was not found')

    image_array = cv.imread(image_path)

    # [INFO] Using the following piece of code results in a 'None' in the training set
    # if image_array == None:
    #     pass
    if channels == 1:
        image_array = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)
    if resized_img_size is not None:
        image_array = cv.resize(image_array, (resized_img_size,resized_img_size))
    return image_array

def get_classes_from_dir(DIR):
    if len(os.listdir(DIR)) == 0:
        raise ValueError('[ERROR] The specified directory does not seem to have any folders in it')
    else:
        classes = [i for i in os.listdir(DIR)]
        return classes
        

def saveNumpy(name, x):
    """
    Saves an array to a .npy file
    Converts to Numpy (if not already)
    """

    x = np.array(x)
    if '.npy' in name:
        np.save(name, x)
    elif '.npz' in name:
        np.savez_compressed(name, x)

def train_val_split(X,y,val_ratio=.2):
    """
    Do not use if mean subtraction is being employed
    Returns X_train, X_val, y_train, y_val
    """
    try:
        from sklearn.model_selection import train_test_split    
        X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=val_ratio,random_state = 2)
        return X_train, X_val, y_train, y_val
    except ModuleNotFoundError:
        print('[ERROR] The Sklearn Python package needs to be installed')


def sort_dict(unsorted_dict, descending=False):
    """
    Sorts a dictionary in ascending order (if descending = False) or descending order (if descending = True)
    """
    return sorted(unsorted_dict.items(), key=lambda x:x[1], reverse=descending)

def plotAcc(histories):
    """
    Plots the model accuracies as 2 graphs
    """
    import matplotlib.pyplot as plt 
    acc = histories.history['acc']
    val_acc = histories.history['val_acc']
    loss = histories.history['loss']
    val_loss = histories.history['val_loss']

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

# -------------------- OPENCV IMAGE-SPECIFIC METHODS --------------------- 

def translate(image, x, y):
    # Defines the translation matrix and performs the translation
    matrix = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv.warpAffine(image, matrix, (image.shape[1], image.shape[0]))

    return shifted

def rotate(image, angle, centre=None, scale=1.0):
    # Grabs the dimensions of the image
    (height, width) = image.shape[:2]

    # If no centre is specified, we grab the centre coordinates of the image
    if centre is None:
        centre = (width // 2, height // 2)

    # Rotates the image
    M = cv.getRotationMatrix2D(centre, angle, scale)
    rotated = cv.warpAffine(image, M, (width, height))

    return rotated

def rotate_bound(image, angle):
    # Grabs the dimensions of the image and then determines the centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Adjusts the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # Performs the actual rotation and returns the image
    return cv.warpAffine(image, M, (nW, nH))

def resize(image, width=None, height=None, interpolation=cv.INTER_AREA):
    """
    Resizes the image while maintaing the aspect ratio of the original image
    """
    # Grabs the image dimensions 
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, we return the original image
    if width is None and height is None:
        return image

    # If  width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # If height is None
    else:
        # Calculates the ratio of the width and constructs the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Resizes the image
    resized = cv.resize(image, dim, interpolation=interpolation)

    return resized

def toMatplotlib(image):
    """
    Converts BGR image ordering to RGB ordering
    """
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)

def url_to_image(url):
    # Converts the image to a Numpy array and reads it in OpenCV
    response = urlopen(url)
    image = np.asarray(bytearray(response.read()), dtype="uint8")
    image = cv.imdecode(image, cv.IMREAD_COLOR)

    return image

def canny(image, sigma=0.33):
    # computes the median of the single channel pixel intensities
    med = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * med))
    upper = int(min(255, (1.0 + sigma) * med))
    edges = cv.Canny(image, lower, upper)

    return edges