"""
Contains functions to compute error metrics.
"""
import numpy as np
from math import sqrt, pow
from sklearn.metrics import classification_report

Z_95 = 1.96


def variance(x, y, model):
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
    var = 0.0
    y_classes = [np.argmax(y_elem, axis=None, out=None) for y_elem in y]

    for idx in range(training_examples):
        if y_classes[idx] == pred_labels[idx]:
            var += (mean - 1).__pow__(2)
        else:
            var += (mean).__pow__(2)

    var = (1/(training_examples-1))*var
    return var


def get_error_bound(x, y, model):
    """
    Computes and prints the error bound within a certain confidence interval for the accuracy of the model.

    :param x: A list of inputs to the model.
    :param y: The target labels. A ndarray.
    :param model: The model in question.
    :return: Nothing
    """

    var = variance(x, y, model)
    training_examples = len(y)
    bound = Z_95*sqrt(var/training_examples)
    print(bound)
    return bound


def get_precision_recall(x, y, model):
    """
    Prints the classification report.

    :param x: A list of inputs to the model.
    :param y: The target labels. A ndarray.
    :param model: The model in question.
    :return: the classification report, a string
    """

    pred = model.predict(x)
    predicted = np.argmax(pred, axis=-1)
    report = classification_report(np.argmax(y, axis=-1), predicted, digits=4)
    print(report)
    return report
