"""
The neural network is built here, using Keras with a TensorFlow backend.
"""

from keras.optimizers import adam
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Convolution2D, Flatten, Dropout
from patchy_san.parameters import FIELD_COUNT, MAX_FIELD_SIZE, CHANNEL_COUNT, CLASS_COUNT


def build_model(learning_rate=0.005, activations="sigmoid"):
    """
    Builds the patchy-san convolutional neural network architecture using Keras.
    The architecture has been chosen arbitrarily, but will be refined later on.
    Currently only data from node properties is considered, but edge data will be
    incorporated into the model later.

    :return:
    """
    input_shape = (FIELD_COUNT*MAX_FIELD_SIZE, CHANNEL_COUNT, 1)

    model = Sequential()
    model.add(Convolution2D(activation='relu', filters=8, kernel_size=(1, 2), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(CLASS_COUNT, activation=activations))
    optimiser = adam(lr=learning_rate)
    model.compile(loss='mean_squared_error',
                  optimizer=optimiser,
                  metrics=['accuracy'])
    return model
