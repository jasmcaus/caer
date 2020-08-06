from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, MaxPooling2D, Conv2D
from preprocess import preprocess
from LeNet import LeNet 
import numpy as np

DIR = r'F:\Dogs and Cats\Train'
categories = ['Dogs', 'Cats']

train, X, y = preprocess(DIR, categories, 3000, isSave=True)

print(train.shape)
print(X.shape)
print(y.shape)