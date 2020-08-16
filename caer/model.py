# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

# Surpressing Tensorflow Warnings
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

# Importing the necessary packages
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D

# Surpressing Tensorflow Warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

def saveModel(model, base_name, learn_rate ,attempt):
    model.save_weights(f'{base_name}-{learn_rate}-{attempt}.h5')
    model.save(f'{base_name}_{attempt}.h5')

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
    
def createDefaultModel(img_size=224, optimizer='adam', loss='binary_crossentropy'):
    try:
        model = Sequential()
        model.add(Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 1)))
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
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        return model

    except ModuleNotFoundError:
        print('[Error] The Tensorflow Python package needs to be installed')

def LeNet(img_size=224, channels=1):
    """
    Adding some extra code for v0.0.14
    """
    try:
        # Initialize the Model
        model = Sequential()
        input_shape = (img_size,img_size,channels)

        # If 'channels first', update the input_shape
        if backend.image_data_format() == 'channels_first':
            input_shape = (channels, img_size,img_size)
        
        # First set
        model.add(Conv2D(32, (3,3), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Second set
        model.add(Conv2D(64, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Flattening
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))

        model.add(Dense(1, activation="relu")) # Softmax works too if multiple Dense nodes required

        return model
    except ModuleNotFoundError:
        print('[ERROR] The Tensorflow Python package needs to be installed')

def VGG16(img_size=224, channels=1):
    try:
        # Initialize the Model
        model = Sequential()
        input_shape = (img_size,img_size,channels)

        # If 'channels first', update the input_shape
        if backend.image_data_format() == 'channels_first':
            input_shape = (channels, img_size,img_size)
        
       # Block 1
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(
                input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

        # Block 2
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

        # Block 3
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

        # Block 4
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

        # Block 5
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

        # Final Block
        # Flattening
        model.add(Flatten())

        model.add(Dense(4096,activation="relu"))
        model.add(Dense(4096,activation="relu"))

        model.add(Dense(1), activate='sigmoid') # Softmax works too if multiple Dense nodes required

        return model
        
    except ModuleNotFoundError:
        print('[ERROR] The Tensorflow Python package needs to be installed')

def testModel(model,categories):
    pass
    # X_test, y_test = preprocess(array) # y_test will be empty
    # x = np.array(X_test)
    # x = normalize(x)
    # test_datagen = imageDataGenerator()

    # # Plotting
    # columns = 5
    # i=0
    # test_labels = []
    # plt.figure(figsize=(30,30))
    # for batch in test_datagen.flow(x, batch_size=1):
    #     pred = model.predict(batch)
    #     if pred > 0.5:
    #         test_labels.append(str(categories[1]))
    #     else:
    #         test_labels.append(str(categories[0]))
    #     plt.subplot(5/columns+1, columns, i+1)
    #     plt.title(f'This is a {test_labels[i]}')
    #     i += 1
    #     # Displaying the first 10 images
    #     if i%10:
    #         break

    #     plt.show()