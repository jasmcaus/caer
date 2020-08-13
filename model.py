def saveModel(model, base_name, learn_rate ,attempt):
    model.save_weights(f'{base_name}-{learn_rate}-{attempt}.h5')
    model.save(f'{base_name}_{attempt}.h5')

def createModel(img_size=224, optimizer='adam', batch_size=32, loss='binary_crossentropy'):
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Dropout
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
    except:
        print('[Error] The Tensorflow Python package needs to be installed')

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