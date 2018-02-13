"""
Contains functions to load training data from .txt files
"""
import numpy as np


def load_data():
    """
    Loads all data available.

    :return: A tuple of ndarrays, (x,y).
    x has shape (training_examples, field_size*field_count, channel_count, 1)
    y has shape (training_examples, class_count)
    """

    x_1 = np.fromfile("training_data/x_train1.txt")
    # Shape is currently hardcoded until i find a better way to do it
    x_1 = x_1.reshape((378, 3, 5, 1))
    y_1 = np.fromfile("training_data/y_train1.txt")
    y_1 = y_1.reshape((378, 2))

    return x_1, y_1
