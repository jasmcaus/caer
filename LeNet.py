from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # Initialize the Model
        model = Sequential()
        input_shape = (width,height,depth)

        # If 'channels first', update the input_shape
        if backend.image_data_format() == 'channels_first':
            input_shape = (depth, width, height)
        
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

        model.add(Dense(1))
        model.add(Activation('sigmoid')) # Softmax works too if multiple Dense nodes required

        return model