from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, MaxPooling2D, Conv2D
import tensorflow.keras.optimizers as optimizers
from preprocess import preprocess,createModel,train_val_split,normalize,imageDataGenerator, saveModel
from LeNet import LeNet 
import numpy as np

DIR = r'F:\Dogs and Cats\Train'
categories = ['Dogs', 'Cats']

train, X, y = preprocess(DIR, categories, 3000, isSave=True)

# Normalizing the Data
X = normalize(X)

# Splitting Data into Train and Validation
X_train, X_val, y_train, y_val = train_val_split(X,y,val_ratio=.2)


# Creating Variables for Convenience
batch_size = 32
epochs = 64
train_len = len(X_train)
val_len = len(X_val)

# Creating the Model
model = createModel(img_size = 100, batch_size = batch_size)

# Compiling the Model
model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=1e-4), metrics=['accuracy'])

"""
# (OPTIONAL) Creating the Image Generator
train_datagen, val_datagen = imageDataGenerator()

# Creating Image Generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

# Fitting the model
histories = model.fit(train_generator, steps_per_epoch=train_len//batch_size, epochs=epochs, validation_data=val_generator, validation_steps=val_len//batch_size)
"""



