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

    # Patchy-san track
    patchy_san_input = Input(shape=patchy_san_input_shape, name='ps_input')
    ps_conv1 = Convolution2D(
        activation='relu',
        filters=8,
        kernel_size=(1, 2),
        input_shape=patchy_san_input_shape,
        name='ps_conv1'
    )(patchy_san_input)
    ps_maxpool1 = MaxPooling2D(pool_size=(1, 2), name='ps_maxpool1')(ps_conv1)
    ps_dropout1 = Dropout(0.1, name='ps_dropout1')(ps_maxpool1)
    ps_flatten1 = Flatten(name='ps_flatten1')(ps_dropout1)

    # Embedding track
    emb_input = Input(shape=(EMBEDDING_LENGTH*MAX_NODES,), name='emb_input')
    emb_embedding = Embedding(
        VOCAB_SIZE,
        EMBEDDING_DIM,
        input_length=EMBEDDING_LENGTH*MAX_NODES,
        name='emb_embedding'
    )(emb_input)
    emb_flatten = Flatten(name='emb_flatten')(emb_embedding)

    merge = concatenate([ps_flatten1, emb_flatten], name='merge')
    dense1 = Dense(8, activation='relu', name='dense1')(merge)
    dense2 = Dense(8, activation='relu', name='dense2')(dense1)
    dropout1 = Dropout(0.1, name='dropout1')(dense2)
    output = Dense(CLASS_COUNT, activation=activations, name='output')(dropout1)
    model = Model(inputs=[patchy_san_input, emb_input], outputs=output)
    optimiser = adam(lr=learning_rate)
    model.compile(loss='mean_squared_error',
                  optimizer=optimiser,
                  metrics=['accuracy'])
    return model
