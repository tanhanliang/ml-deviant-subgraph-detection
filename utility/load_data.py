"""
Contains functions to load training data from .txt files
"""
import numpy as np


def load_data():
    """
    Loads data for the latest version of the model.

    :return: A tuple of (list of inputs, target values). The inputs and target values
    are numpy ndarrays
    """

    xpn = np.fromfile("training_data/x_patchy_nodes6.txt")
    xpe = np.fromfile("training_data/x_patchy_edges6.txt")
    xe = np.fromfile("training_data/x_embed6.txt")
    y = np.fromfile("training_data/y_train6.txt")

    xpn = xpn.reshape((2000, 6, 5, 1))
    xpe = xpe.reshape((2000, 36, 2, 1))
    xe = xe.reshape((2000, 120))
    y = y.reshape((2000, 2))

    return [xpn, xpe, xe], y
