"""
Contains functions to load training data from .txt files
"""
import numpy as np


def load_data():
    """
    Loads data for the latest version of the model.

    :return: A tuple of ndarrays, (x,y).
    x has shape (training_examples, field_size*field_count, channel_count, 1)
    y has shape (training_examples, class_count)
    """

    x_embed = np.fromfile("training_data/x_embed4.txt")
    x_patchy = np.fromfile("training_data/x_patchy4.txt")
    y = np.fromfile("training_data/y_train4.txt")
    # Shape is currently hardcoded until i find a better way to do it
    x_embed = x_embed.reshape((2000, 40))
    x_patchy = x_patchy.reshape((2000, 4, 5, 1))
    y = y.reshape((2000, 2))

    return x_patchy, x_embed, y
