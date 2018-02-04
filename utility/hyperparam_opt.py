"""
Contains functions to optimize hyperparameters.
"""

from patchy_san.cnn import build_model

LEARNING_RATES = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
MOMENTUM_VALS = [0, 0.5, 0.7, 0.9, 0.95, 0.99]
ACTIVATIONS = ['relu']


def grid_search(x_train, y_train):
    """
    Performs grid search using the values specified.
    :param x_train: training data in the form of a NumPy array
    :param y_train: target data in the form of a Numpy array
    :return: A tuple describing the best hyperparams found:
    (best_rate, best_momentum, best_activation, best_accuracy)
    """
    best_rate = -1
    best_momentum = -1
    best_activation = ""
    best_accuracy = 0
    count = 0

    for rate in LEARNING_RATES:
        for momentum in MOMENTUM_VALS:
            for activation in ACTIVATIONS:
                model = build_model(rate, momentum, activation)
                model.fit(x_train,
                          y_train,
                          epochs=100,
                          batch_size=5,
                          validation_split=0.0,
                          shuffle=True)

                accuracy = model.evaluate(x_train, y_train)[1]
                count += 1
                print("##################COUNT = " + str(count))

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_activation = activation
                    best_momentum = momentum
                    best_rate = rate

    return best_rate, best_momentum, best_activation, best_accuracy
