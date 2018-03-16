"""
The neural network is built here, using Keras with a TensorFlow backend.
"""

from keras.optimizers import adam, RMSprop
from keras.models import Model, Sequential
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

    :return: A keras Model
    """
    ps_nodes_input_shape = (FIELD_COUNT*MAX_FIELD_SIZE, CHANNEL_COUNT, 1)
    ps_edges_input_shape = (FIELD_COUNT*MAX_NODES*MAX_NODES, 2, 1)

    # Patchy-san nodes track
    ps_nodes_input = Input(shape=ps_nodes_input_shape, name='ps_nodes_input')
    psn_conv1 = Convolution2D(
        activation='relu',
        filters=8,
        kernel_size=(1, 2),
        input_shape=ps_nodes_input_shape,
        name='ps_nodes_conv1'
    )(ps_nodes_input)
    psn_maxpool1 = MaxPooling2D(pool_size=(1, 2), name='ps_nodes_maxpool1')(psn_conv1)
    psn_dropout1 = Dropout(0.1, name='ps_nodes_dropout1')(psn_maxpool1)
    psn_flatten1 = Flatten(name='ps_nodes_flatten1')(psn_dropout1)

    # Patchy-san edges track
    ps_edges_input = Input(shape=ps_edges_input_shape, name='ps_edges_input')
    pse_conv1 = Convolution2D(
        activation='relu',
        filters=8,
        kernel_size=(1, 2),
        input_shape=ps_edges_input_shape,
        name='ps_edges_conv1'
    )(ps_edges_input)
    # pse_maxpool1 = MaxPooling2D(pool_size=(1, 2), name='ps_edges_maxpool1')(pse_conv1)
    # pse_dropout1 = Dropout(0.1, name='ps_edges_dropout1')(pse_maxpool1)
    pse_dropout1 = Dropout(0.1, name='ps_edges_dropout1')(pse_conv1)
    pse_flatten1 = Flatten(name='ps_edges_flatten1')(pse_dropout1)

    # Embedding track
    emb_input = Input(shape=(EMBEDDING_LENGTH*MAX_NODES*2,), name='emb_input')
    emb_embedding = Embedding(
        VOCAB_SIZE,
        EMBEDDING_DIM,
        input_length=EMBEDDING_LENGTH*MAX_NODES*2,
        name='emb_embedding'
    )(emb_input)
    emb_flatten = Flatten(name='emb_flatten')(emb_embedding)

    merge = concatenate([psn_flatten1, pse_flatten1, emb_flatten], name='merge')
    dense1 = Dense(24, activation='relu', name='dense1')(merge)
    dense2 = Dense(24, activation='relu', name='dense2')(dense1)
    dropout1 = Dropout(0.1, name='dropout1')(dense2)
    output = Dense(CLASS_COUNT, activation=activations, name='output')(dropout1)
    model = Model(inputs=[ps_nodes_input, ps_edges_input, emb_input], outputs=output)
    optimiser = adam(lr=learning_rate)
    model.compile(loss='mean_squared_error',
                  optimizer=optimiser,
                  metrics=['accuracy'])
    return model


def build_double_input_model(learning_rate=0.005, activations="sigmoid"):
    """
    Builds the model which only implemented patchy-san for nodes and word embeddings.
    Also for testing purposes.

    :param learning_rate:
    :param activations:
    :return: A keras Model
    """

    ps_nodes_input_shape = (FIELD_COUNT*MAX_FIELD_SIZE, CHANNEL_COUNT, 1)

    # Patchy-san nodes track
    ps_nodes_input = Input(shape=ps_nodes_input_shape, name='ps_nodes_input')
    psn_conv1 = Convolution2D(
        activation='relu',
        filters=8,
        kernel_size=(1, 2),
        input_shape=ps_nodes_input_shape,
        name='ps_nodes_conv1'
    )(ps_nodes_input)
    psn_maxpool1 = MaxPooling2D(pool_size=(1, 2), name='ps_nodes_maxpool1')(psn_conv1)
    psn_dropout1 = Dropout(0.1, name='ps_nodes_dropout1')(psn_maxpool1)
    psn_flatten1 = Flatten(name='ps_nodes_flatten1')(psn_dropout1)

    # Embedding track
    emb_input = Input(shape=(EMBEDDING_LENGTH*MAX_NODES*2,), name='emb_input')
    emb_embedding = Embedding(
        VOCAB_SIZE,
        EMBEDDING_DIM,
        input_length=EMBEDDING_LENGTH*MAX_NODES*2,
        name='emb_embedding'
    )(emb_input)
    emb_flatten = Flatten(name='emb_flatten')(emb_embedding)

    merge = concatenate([psn_flatten1, emb_flatten], name='merge')
    dense1 = Dense(8, activation='relu', name='dense1')(merge)
    dense2 = Dense(8, activation='relu', name='dense2')(dense1)
    dropout1 = Dropout(0.1, name='dropout1')(dense2)
    output = Dense(CLASS_COUNT, activation=activations, name='output')(dropout1)
    model = Model(inputs=[ps_nodes_input, emb_input], outputs=output)
    optimiser = adam(lr=learning_rate)
    model.compile(loss='mean_squared_error',
                  optimizer=optimiser,
                  metrics=['accuracy'])
    return model


def build_single_input_model(learning_rate=0.005, activations="sigmoid"):
    """
    Builds the old, single-input model without word embeddings. This is here so that
    I can run experiments on the old model.

    The training data generation framework may not support generation of training data
    for this old model, but you can retrieve stored training data. Remember to reshape
    it correctly, as described in the 'training_data/about_training_data.txt' file!

    :param learning_rate: A float
    :param activations: A string. The last layer's activation function
    :return: A keras Model
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
