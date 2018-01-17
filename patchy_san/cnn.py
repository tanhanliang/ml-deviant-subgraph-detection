"""
The neural network is built here, using Keras with a TensorFlow backend.
"""

from keras.models import Sequential
from keras.layers import Dense, MaxPooling1D, Convolution1D, Flatten, Dropout
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
    input_shape = (FIELD_COUNT, MAX_FIELD_SIZE, CHANNEL_COUNT)

    model = Sequential()
    model.add(Convolution1D(16, 3, 3, activation='relu', input_shape=input_shape))
    model.add(Convolution1D(16, 3, 3, activation='relu'))
    model.add(MaxPooling1D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# TODO: function to train model
