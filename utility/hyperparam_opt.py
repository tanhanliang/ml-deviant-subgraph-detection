"""
Contains functions to optimize hyperparameters.
"""

from patchy_san.cnn import build_model
from sklearn.model_selection import StratifiedKFold
import numpy as np

LEARNING_RATES = [1, 0.5, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005]
# MOMENTUM_VALS = [0, 0.5, 0.7, 0.9, 0.99]
ACTIVATIONS = ['relu', 'sigmoid', 'softmax']
VALIDATION_SPLIT = 0.2


def grid_search(x_train, y_train):
    """
    Performs grid search using the values specified.
    :param x_train: training data in the form of a NumPy array
    :param y_train: target data in the form of a Numpy array
    :return: A tuple describing the best hyperparams found:
    (best_rate, best_momentum, best_activation, best_accuracy)
    """
    best_rate = -1
    best_activation = ""
    best_accuracy = 0
    count = 0
    training_examples = int(x_train.shape[0]*(1-VALIDATION_SPLIT))

    for rate in LEARNING_RATES:
        for activation in ACTIVATIONS:
            model = build_model(rate, activation)
            model.fit(x_train,
                      y_train,
                      epochs=100,
                      batch_size=5,
                      validation_split=VALIDATION_SPLIT,
                      shuffle=True)

            # Evaluate model on last 20% of data which was not seen by model
            accuracy = model.evaluate(x_train[training_examples:], y_train[training_examples:])[1]
            count += 1
            print("##################COUNT = " + str(count))

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_activation = activation
                best_rate = rate

    return best_rate, best_activation, best_accuracy


def cross_validation(inputs, y, folds, epochs, learning_rate, activation):
    """
    Performs k-fold cross validation for a particular dataset.

    :param inputs: The training data for each input to the model, in a list
    :param y: target data in the form of a Numpy array
    :param epochs: An integer
    :param folds: An integer
    :param learning_rate: A float
    :param activation: A string
    :return: average accuracy and loss as a tuple
    """

    y_labels = np.argmax(y, axis=1)
    skf = StratifiedKFold(n_splits=folds)
    idx = 1
    average_accuracy = 0
    average_loss = 0

    for train_indices, test_indices in skf.split(inputs[0], y_labels):
        print("Training on fold " + str(idx))
        train = []
        test = []

        for inpt in inputs:
            train.append(inpt[train_indices])
            test.append(inpt[test_indices])

        y_train, y_test = y[train_indices], y[test_indices]

        model = build_model(learning_rate, activation)
        model.fit(train, y_train, epochs=epochs, batch_size=10, validation_split=0.0, shuffle=True)
        loss, accuracy = model.evaluate(test, y_test)
        average_accuracy += accuracy
        average_loss += loss
        print("Accuracy for the " + str(idx) + "th fold: " + str(accuracy))
        idx += 1

    average_accuracy /= folds
    average_loss /= folds
    print("Average accuracy: " + str(average_accuracy))
    print("Average loss: " + str(average_loss))
    return average_accuracy, average_loss
