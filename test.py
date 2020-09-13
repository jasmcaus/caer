#pylint: disable=bare-except

import cv2 as cv
import numpy as np
import os

file = r'C:\Users\aus\Downloads'
train = []

def readImg(image_path, IMG_SIZE, channels=1):
    try:
        image_array = cv.imread(image_path)

        # [INFO] Using the following piece of code results in a 'None' in the training set
        # if image_array == None:
        #     pass
        if channels == 1:
            image_array = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)

        image_array = cv.resize(image_array, (IMG_SIZE,IMG_SIZE))
        return image_array
    except:
        return None

for img in os.listdir(file):
    if img.endswith('.png') or img.endswith('.jpg'):
        path = os.path.join(file, img)
        img = readImg(path, 300, 1)
        img = img.astype('float32') - 80

        train.append(img)

print('Length=', len(train))
np.save('train_float_subtracted_gray.npy', train)