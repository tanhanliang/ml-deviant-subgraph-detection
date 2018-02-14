"""
Contains functions to format the training data into ndarrays that can be used to train the
model.
"""
from patchy_san.make_cnn_input import build_groups_of_receptive_fields, build_tensor_naive_hashing
from make_training_data.filter_training_data import get_training_data
from patchy_san.parameters import FIELD_COUNT, MAX_FIELD_SIZE, CHANNEL_COUNT, CLASS_COUNT
import numpy as np


def format_all_training_data():
    """
    Queries the database for data according to several predefined rules, then processes
    them into two ndarrays.

    :return: A tuple (x_data,y_target) of ndarrays. x_data has shape (s,field_count,max_field_size,n)
    and y_target has shape (s,1)
    s: number of training samples
    n: number of attributes
    """
    import time
    start = time.time()
    x_data_list = []
    y_target_list = []

    training_data = get_training_data()

    for (label, training_nodes, training_edges) in training_data:
        receptive_fields_groups = build_groups_of_receptive_fields(training_nodes, training_edges)
        # For training data for most classes there should only be one receptive field group
        for fields_list in receptive_fields_groups:
            training_example_tensor = build_tensor_naive_hashing(fields_list)
            x_data_list.append(training_example_tensor)
            y_target_list.append(label)

    training_examples = len(x_data_list)
    x_data = np.ndarray((training_examples, FIELD_COUNT, MAX_FIELD_SIZE, CHANNEL_COUNT))
    y_target = np.asarray(y_target_list, dtype=np.int32)

    idx = 0
    while idx < training_examples:
        x_data[idx] = x_data_list[idx]
        idx += 1
    end = time.time()
    print("Time elapsed(seconds): "+str(end-start))
    return x_data, y_target


def create_balanced_training_set(x_data, y_target, limit):
    """
    Ensure that training set contains equal numbers of training examples for each class.

    :param x_data: A 4D NumPy ndarray (training_examples, FIELD_COUNT, MAX_FIELD_SIZE, CHANNEL_COUNT)
    :param y_target: A 1D NumPy ndarray (training_examples)
    :param limit: An integer which represents the max training examples for each class.
    :return: A tuple of ndarrays
    """

    class_counts = [0 for _ in range(CLASS_COUNT)]
    x_train = []
    y_train = []

    for i in range(len(x_data)):
        label = y_target[i]
        if class_counts[label] < limit:
            class_counts[label] += 1
            x_train.append(x_data[i])
            y_train.append(y_target[i])

    new_x = np.ndarray((len(x_train), FIELD_COUNT, MAX_FIELD_SIZE, CHANNEL_COUNT))
    new_y = np.ndarray((len(y_train),))

    for i in range(len(x_train)):
        new_x[i] = x_train[i]
        new_y[i] = y_train[i]

    return new_x, new_y


def reshape_training_data(x_train):
    """
    Reshapes 3D tensors into a 2D tensor by combining the first two dimensions.
    The final tensor has 4 dimensions specified, with the last dimension set as 1.

    TODO: Fix the final dimension.

    :param x_train: A ndarray with shape (s,field_count,max_field_size,n)
    :return: A ndarray. It has shape (training examples,field_count*max_field_size,n, 1)
    """

    new_shape = (x_train.shape[0], FIELD_COUNT*MAX_FIELD_SIZE, CHANNEL_COUNT, 1)
    return x_train.reshape(new_shape)


def shuffle_datasets(x_train, y_train):
    """
    Shuffles the provided training datasets and labels together, along the first axis

    :param x_train: A ndarray
    :param y_train: A ndarray
    :return: A tuple of shuffled ndarrays
    """

    permutation = np.random.permutation(x_train.shape[0])
    return x_train[permutation], y_train[permutation]


def get_final_datasets():
    """
    Gets and formats the datasets into a form ready to be fed to the model.

    :return: A tuple of ndarrays (x_new, y_new). x_new has dimensions
    (training_samples, field_cound*max_field_size, channel_count)
    y_new has dimensions (training_samples, number_of_classes)
    """

    x, y = format_all_training_data()
    _, counts = np.unique(y, return_counts=True)

    if len(counts) == 1:
        return np.ndarray((0,)), np.ndarray((0,))

    min_count = np.amin(counts)
    x_train, y_train = create_balanced_training_set(x, y, min_count)

    from keras.utils import to_categorical
    y_new = to_categorical(y_train)
    x_new = reshape_training_data(x_train)

    return shuffle_datasets(x_new, y_new)
