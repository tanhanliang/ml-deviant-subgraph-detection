"""
The neural network is built here, using Keras with a TensorFlow backend.
"""

from keras.optimizers import adam
from keras.models import Model
from keras.layers import Dense, MaxPooling2D, Convolution2D, Flatten, Dropout, Input, Embedding
from patchy_san.parameters import FIELD_COUNT, MAX_FIELD_SIZE, CHANNEL_COUNT, CLASS_COUNT
from patchy_san.parameters import EMBEDDING_LENGTH, EMBEDDING_DIM, MAX_NODES, VOCAB_SIZE
from keras.layers.merge import concatenate


def build_model(learning_rate=0.005, activations="sigmoid"):
    """
    Builds the patchy-san convolutional neural network architecture using Keras.
    The architecture has been chosen arbitrarily, but will be refined later on.
    Currently only data from node properties is considered, but edge data will be
    incorporated into the model later.

    :return:
    """
    patchy_san_input_shape = (FIELD_COUNT*MAX_FIELD_SIZE, CHANNEL_COUNT, 1)

    patchy_san_input = Input(shape=patchy_san_input_shape)
    conv = Convolution2D(
            activation='relu',
            filters=8,
            kernel_size=(1, 2),
            input_shape=patchy_san_input_shape)(patchy_san_input)
    maxpool = MaxPooling2D(pool_size=(1, 2))(conv)
    dropout1 = Dropout(0.1)(maxpool)
    flatten1 = Flatten()(dropout1)

    embedding_input = Input(shape=(EMBEDDING_LENGTH*MAX_NODES,))
    embedding = Embedding(VOCAB_SIZE,
                          EMBEDDING_DIM,
                          input_length=EMBEDDING_LENGTH*MAX_NODES)(embedding_input)
    flatten2 = Flatten()(embedding)

    merge = concatenate([flatten1, flatten2])
    dense1 = Dense(8, activation='relu')(merge)
    dense2 = Dense(8, activation='relu')(dense1)
    dropout2 = Dropout(0.1)(dense2)
    output = Dense(CLASS_COUNT, activation=activations)(dropout2)
    model = Model(inputs=[patchy_san_input, embedding_input], outputs=output)
    optimiser = adam(lr=learning_rate)
    model.compile(loss='mean_squared_error',
                  optimizer=optimiser,
                  metrics=['accuracy'])
    return model
