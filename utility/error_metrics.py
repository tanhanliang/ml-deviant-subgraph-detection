"""
Contains functions to compute error metrics.
"""
import numpy as np
from math import sqrt, pow

Z_95 = 1.96


def std_dev(x, y, model):
    """
    Computes the estimated standard deviation using Bessel's correction.

    :param x: A list of inputs to the model.
    :param y: The target labels. A ndarray.
    :param model: The model in question.
    :return: A float, the std deviation.
    """

    probabilities = model.predict(x)
    pred_labels = probabilities.argmax(axis=-1)
    training_examples = len(pred_labels)

    # Accuracy
    mean = model.evaluate(x, y)[1]
    std = 0.0
    y_classes = [np.argmax(y_elem, axis=None, out=None) for y_elem in y]

    for idx in range(training_examples):
        if y_classes[idx] == pred_labels[idx]:
            std += (mean - 1).__pow__(2)
        else:
            std += (mean).__pow__(2)

    std = (1/(training_examples-1))*std
    return std


def compute_error_bound(x, y, model):
    """
    Computes the error bound within a certain confidence interval for the accuracy of the model.

    :param x: A list of inputs to the model.
    :param y: The target labels. A ndarray.
    :param model: The model in question.
    :return: A float, the error bound.
    """

    std = std_dev(x, y, model)
    training_examples = len(y)
    return Z_95*sqrt(pow(std, 2)/training_examples)
