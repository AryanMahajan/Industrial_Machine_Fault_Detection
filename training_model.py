from loading_data import create_training_data

import tensorflow as tf

#importing training and test data
x_train, y_train = create_training_data()


x_train = x_train.reshape(-1, 50, 50, 1)


def training_model():
    
    #initialising teh callback class
    callbacks = myCallback()


    #building model
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding = 'same', input_shape=x_train.shape[1:]))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(tf.keras.layers.Dense(64, activation = 'relu'))

    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation('sigmoid'))
    
    #model Compilation
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=["accuracy"])

    #Fitting the model
    history = model.fit(x_train,y_train, batch_size=32, validation_split = 0.1, epochs = 10)

    #Saving the model
    model.save("model.keras")

    return model, history