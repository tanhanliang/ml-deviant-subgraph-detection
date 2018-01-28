"""
The neural network is built here, using Keras with a TensorFlow backend.
"""

from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Convolution2D, Flatten, Dropout
from patchy_san.parameters import FIELD_COUNT, MAX_FIELD_SIZE, CHANNEL_COUNT


def build_model():
    """
    Builds the patchy-san convolutional neural network architecture using Keras.
    The architecture has been chosen arbitrarily, but will be refined later on.
    Currently only data from node properties is considered, but edge data will be
    incorporated into the model later.

    :param input_shape: A tuple containing the (height, width, channels) of the input
    :return:
    """
    input_shape = (FIELD_COUNT*MAX_FIELD_SIZE, CHANNEL_COUNT, 1)

    model = Sequential()
    model.add(Convolution2D(activation='relu', filters=8, kernel_size=(1,2), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(1,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# TODO: function to train model
