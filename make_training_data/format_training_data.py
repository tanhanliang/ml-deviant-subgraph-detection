"""build_in_out_edges
Contains functions to format the training data into ndarrays that can be used to train the
model.
"""
from patchy_san.make_cnn_input import build_groups_of_receptive_fields, build_tensor_naive_hashing
from patchy_san.make_cnn_input import build_embedding
from patchy_san.parameters import FIELD_COUNT, MAX_FIELD_SIZE, CHANNEL_COUNT, CLASS_COUNT, EMBEDDING_LENGTH
from patchy_san.parameters import MAX_NODES
from data_processing.preprocessing import get_graphs_by_result
import numpy as np


def label_and_process_data(results):
    """
    Given a list of BoltStatementResults, each of which corresponds to training data for
    one class, process it by labelling it correctly and creating graphs for each training
    example (e.g one BoltStatementResult has many training examples).

    :param results A list of BoltstatementResults.
    :return: A list of tuplesof (label, graph). label is an integer, graph is a Graph object/
    """

    training_data = []
    label = 0

    for result in results:
        graph_list = get_graphs_by_result(result)

        for graph in graph_list:
            training_data.append((label, graph))

        label += 1

    print("Raw data has been formatted into Graph objects.")
    return training_data


def format_all_training_data(training_data):
    """
    Queries the database for data according to several predefined rules, then processes
    them into two ndarrays.

    :param training_data:A list of tuples (label, graph). label is an integer,
    graph is a Graph object.
    :return: A tuple (x_patchy_input, x_embedding_input, y_target). The first argument is
    the input ndarray created by patchy_san, the second is the ndarray created by word
    embeddings. The last, y_target is also an ndarray.

    x_patchy_input has shape (training_examples,field_count,max_field_size,channels)
    x_embedding_input (training_examples, max_nodes_in_input_graph*embedding_length*2)
    y_target has shape (training_examples, 1)
    """
    import time
    start = time.time()
    x_data_list = []
    y_target_list = []

    print("Processing training graphs into tensors...")
    for (label, graph) in training_data:
        receptive_fields_groups = build_groups_of_receptive_fields(graph)

        # For training data there will only be one receptive field group, so assume
        # that length of receptive_field_groups is 1
        if len(receptive_fields_groups) != 1:
            msg = "More or less than one receptive field group exists in the training example."
            raise ValueError(msg)

        training_example_tensor = build_tensor_naive_hashing(receptive_fields_groups[0])
        embedding = build_embedding(graph)
        x_data_list.append((training_example_tensor, embedding))
        y_target_list.append(label)

    training_examples = len(x_data_list)
    x_patchy_input = np.ndarray((training_examples, FIELD_COUNT, MAX_FIELD_SIZE, CHANNEL_COUNT))
    x_embedding_input = np.ndarray((training_examples, MAX_NODES*EMBEDDING_LENGTH*2))
    y_target = np.asarray(y_target_list, dtype=np.int32)

    idx = 0
    while idx < training_examples:
        x_patchy_input[idx] = x_data_list[idx][0]
        x_embedding_input[idx] = x_data_list[idx][1]
        idx += 1

    end = time.time()
    print("Time elapsed to process training graphs into tensors (seconds): "+str(end-start))
    return x_patchy_input, x_embedding_input, y_target


def create_balanced_training_set(x_patchy_input, x_embedding_input, y_target, limit):
    """
    Ensure that training set contains equal numbers of training examples for each class.

    :param x_patchy_input: A ndarray with shape (training_examples,field_count,max_field_size,channels)
    :param x_embedding_input: A ndarray with shape (training_examples, MAX_NODES*EMBEDDING_LENGTH*2)
    :param y_target: A 1D NumPy ndarray (training_examples,)
    :param limit: An integer which represents the max training examples for each class.
    :return: A tuple of ndarrays
    """

    class_counts = [0 for _ in range(CLASS_COUNT)]
    new_x_patchy_input = np.zeros((limit*CLASS_COUNT, FIELD_COUNT, MAX_FIELD_SIZE, CHANNEL_COUNT))
    new_x_embedding_input = np.zeros((limit*CLASS_COUNT, EMBEDDING_LENGTH*MAX_NODES*2))
    new_y = np.ndarray((limit*CLASS_COUNT,))
    idx = 0

    for i in range(len(y_target)):
        label = y_target[i]
        if class_counts[label] < limit:
            class_counts[label] += 1
            new_x_patchy_input[idx] = x_patchy_input[i]
            new_x_embedding_input[idx] = x_embedding_input[i]
            new_y[idx] = y_target[i]
            idx += 1

    return new_x_patchy_input, new_x_embedding_input, new_y


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


def shuffle_datasets(x_patchy, x_embedding, y_train):
    """
    Shuffles the provided training datasets and labels together, along the first axis

    :param x_patchy: A ndarray with shape (training_examples,field_count,max_field_size,channels)
    :param x_embedding: A ndarray with shape (training_examples, MAX_NODES*EMBEDDING_LENGTH*2)

    :param y_train: A ndarray
    :return: A tuple of shuffled ndarrays
    """

    permutation = np.random.permutation(y_train.shape[0])
    return x_patchy[permutation], x_embedding[permutation], y_train[permutation]


def process_training_examples(training_graphs):
    """
    Gets and formats the datasets into a form ready to be fed to the model.

    :param training_graphs:A list of tuples (label, graph). label is an integer,
    graph is a Graph object.
    :return: A tuple of ndarrays (x_new, y_new). x_new has dimensions
    (training_samples, field_cound*max_field_size, channel_count)
    y_new has dimensions (training_samples, number_of_classes)
    """

    x_patchy, x_embedding, y = format_all_training_data(training_graphs)
    _, counts = np.unique(y, return_counts=True)

    if len(counts) == 1:
        raise ValueError("No training data has been created. Pattern not found.")

    min_count = np.amin(counts)
    x_patchy, x_embedding, y_train = create_balanced_training_set(x_patchy, x_embedding, y, min_count)
    print("The training data has been balanced.")

    from keras.utils import to_categorical
    y_new = to_categorical(y_train)
    x_patchy_new = reshape_training_data(x_patchy)

    return shuffle_datasets(x_patchy_new, x_embedding, y_new)


def get_final_datasets(results):
    """
    Given a list of BoltStatementResults, each corresponding to training data for one
    training pattern, generates training data and formats it properly.

    :param results: A list of BoltStatementResults
    :return: A tuple of ndarrays (x_new, y_new). x_new has dimensions
    (training_samples, field_cound*max_field_size, channel_count)
    y_new has dimensions (training_samples, number_of_classes)
    """

    training_graphs = label_and_process_data(results)
    return process_training_examples(training_graphs)
