"""
Contains functions to help me debug the cnn.
"""

import keras.backend as backend


def get_nth_layer_output_fn(model, n):
    """
    Returns a Keras function to return the output of a certain layer. Similar to the
    original Keras model, this function takes a tuple of numpy ndarrays as input:
    (x,y). x is a 4D numpy array of training examples, y is a 2D numpy array of
    target values per training example using one-hot encoding.

    To use the resulting function:

    layer3_output = get_nth_layer_output_fn(model, 3)
    output = layer3_output([x_val, learning_phase_flag])

    learning_phase_flag is 0 for test mode, 1 for training mode.

    :param model: The Keras model
    :param n: The layer that you want output from
    :return: Keras function that will return the output of a certain layer given a certain input
    """

    return backend.function([model.layers[0].input, backend.learning_phase()], [model.layers[n].output])
