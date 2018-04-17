"""build_in_out_edges
Contains functions to format the training data into ndarrays that can be used to train the
model.
"""
import patchy_san.make_cnn_input as make_input
import patchy_san.parameters as params
import data_processing.preprocessing as preprocess

import numpy as np


def label_and_process_data(results):
    """
    Given a list of BoltStatementResults, each of which corresponds to training data for
    one class, process it by labelling it correctly and creating graphs for each training
    example (e.g one BoltStatementResult has many training examples).

    :param results A list of BoltstatementResults.
    :return: A list of tuplesof (label, graph). label is an integer, graph is a Graph object/
    """

    training_graphs = []
    label = 0

    for result in results:
        graph_list = preprocess.get_graphs_by_result(result)

        for graph in graph_list:
            training_graphs.append((label, graph))

        label += 1

    print("Raw data has been formatted into Graph objects.")
    return training_graphs


def format_all_training_data(training_graphs):
    """
    Queries the database for data according to several predefined rules, then processes
    them into two ndarrays.

    :param training_graphs:A list of tuples (label, graph). label is an integer,
    graph is a Graph object.
    :return: A tuple (x_patchy_nodes, x_patchy_edges, x_embedding_input, y_target).
    The first argument is the input ndarray created by patchy_san for nodes, the second is
    the ndarray created by patchy_san for edges, and the third is the ndarray created by word
    embeddings. The last, y_target is also an ndarray.

    x_patchy_nodes has shape (training_examples,field_count*max_field_size,CHANNELS, 1)
    x_patchy_edges has shape (training_examples,field_count*max_field_size*max_field_size,2)
    x_embedding_input (training_examples, max_nodes_in_input_graph*embedding_length*2)
    y_target has shape (training_examples, 1)
    """
    import time
    start = time.time()
    x_data_list = []
    y_target_list = []

    print("Processing training graphs into tensors...")
    for (label, graph) in training_graphs:
        receptive_fields_groups = make_input.build_groups_of_receptive_fields(graph)

        # For training data there will only be one receptive field group, so assume
        # that length of receptive_field_groups is 1
        if len(receptive_fields_groups) != 1:
            msg = "More or less than one receptive field group exists in the training example."
            raise ValueError(msg)

        nodes_tensor = make_input.build_tensor_naive_hashing(receptive_fields_groups[0])
        edges_tensor = make_input.build_edges_tensor(receptive_fields_groups[0])
        embedding = make_input.build_embedding(graph)
        x_data_list.append((nodes_tensor, edges_tensor, embedding))
        y_target_list.append(label)

    training_examples = len(x_data_list)

    assert(training_examples > 0)

    train_patchy_nodes_shape = (training_examples,) + x_data_list[0][0].shape
    train_patchy_edges_shape = (training_examples,) + x_data_list[0][1].shape
    train_embed_shape = (training_examples,) + x_data_list[0][2].shape

    # train_patchy_nodes_shape = (
    #     training_examples,
    #     params.FIELD_COUNT*params.MAX_FIELD_SIZE,
    #     params.CHANNEL_COUNT, 1
    # )
    # train_patchy_edges_shape = (
    #     training_examples,
    #     params.MAX_NODES*params.MAX_NODES*params.FIELD_COUNT,
    #     params.EDGE_PROP_COUNT, 1
    # )
    # train_embed_shape = (training_examples, params.MAX_NODES*params.EMBEDDING_LENGTH*2)

    x_patchy_nodes = np.ndarray(train_patchy_nodes_shape)
    x_patchy_edges = np.ndarray(train_patchy_edges_shape)
    x_embedding_input = np.ndarray(train_embed_shape)
    y_target = np.asarray(y_target_list, dtype=np.int32)

    idx = 0
    while idx < training_examples:
        x_patchy_nodes[idx] = x_data_list[idx][0]
        x_patchy_edges[idx] = x_data_list[idx][1]
        x_embedding_input[idx] = x_data_list[idx][2]
        idx += 1

    end = time.time()
    print("Time elapsed to process training graphs into tensors (seconds): "+str(end-start))
    return x_patchy_nodes, x_patchy_edges, x_embedding_input, y_target


def create_balanced_training_set(x_patchy_nodes, x_patchy_edges, x_embedding_input, y_target, limit):
    """
    Ensure that training set contains equal numbers of training examples for each class.

    :param x_patchy_nodes: A ndarray with shape (training_examples,field_count*max_field_size,channels,1)
    :param x_patchy_edges: A ndarray with shape (training_examples,field_count*max_field_size*max_field_size,2)
    :param x_embedding_input: A ndarray with shape (training_examples, max_field_size*embedding_length*2)
    :param y_target: A 1D NumPy ndarray (training_examples,)
    :param limit: An integer which represents the max training examples for each class.
    :return: A tuple of ndarrays
    """
    patchy_nodes_shape = (limit*params.CLASS_COUNT,) + x_patchy_nodes[0].shape
    patchy_edges_shape = (limit*params.CLASS_COUNT,) + x_patchy_edges[0].shape
    embedding_shape = (limit*params.CLASS_COUNT,) + x_embedding_input[0].shape

    # patchy_nodes_shape = (
    #     limit*params.CLASS_COUNT,
    #     params.FIELD_COUNT*params.MAX_FIELD_SIZE,
    #     params.CHANNEL_COUNT,
    #     1
    # )
    # patchy_edges_shape = (
    #     limit*params.CLASS_COUNT,
    #     params.FIELD_COUNT*params.MAX_NODES*params.MAX_NODES,
    #     params.EDGE_PROP_COUNT,
    #     1
    # )
    # embedding_shape = (limit*params.CLASS_COUNT, params.EMBEDDING_LENGTH*params.MAX_NODES*2)

    class_counts = [0 for _ in range(params.CLASS_COUNT)]
    new_x_patchy_nodes = np.zeros(patchy_nodes_shape)
    new_x_patchy_edges = np.zeros(patchy_edges_shape)
    new_x_embedding_input = np.zeros(embedding_shape)
    new_y = np.ndarray((limit*params.CLASS_COUNT,))
    idx = 0

    for i in range(len(y_target)):
        label = y_target[i]
        if class_counts[label] < limit:
            class_counts[label] += 1
            new_x_patchy_nodes[idx] = x_patchy_nodes[i]
            new_x_patchy_edges[idx] = x_patchy_edges[i]
            new_x_embedding_input[idx] = x_embedding_input[i]
            new_y[idx] = y_target[i]
            idx += 1

    return new_x_patchy_nodes, new_x_patchy_edges, new_x_embedding_input, new_y


def shuffle_datasets(x_patchy_nodes, x_patchy_edges, x_embedding, y_train):
    """
    Shuffles the provided training datasets and labels together, along the first axis

    :param x_patchy_nodes: A ndarray with shape (training_examples,field_count*max_field_size,channels,1)
    :param x_patchy_edges: A ndarray with shape
    (training_examples, field_count*max_field_size*max_field_size, EDGE_PROP_COUNT)
    :param x_embedding: A ndarray with shape (training_examples, MAX_NODES*EMBEDDING_LENGTH*2)
    :param y_train: A ndarray
    :return: A tuple of shuffled ndarrays
    """

    permutation = np.random.permutation(y_train.shape[0])
    return x_patchy_nodes[permutation], x_patchy_edges[permutation], x_embedding[permutation], y_train[permutation]


def process_training_examples(training_graphs):
    """
    Gets and formats the datasets into a form ready to be fed to the model.

    :param training_graphs:A list of tuples (label, graph). label is an integer,
    graph is a Graph object.
    :return: A tuple of ndarrays (x_patchy_nodes, x_patchy_edges, x_embedding, y_new).
    x_patchy_nodes has dimensions (training_samples, field_count*max_field_size, channel_count)
    x_patchy_edges has dimensions (training_samples, field_count*max_field_size*max_field_size, EDGE_PROP_COUNT)
    x_embedding has dimensions (training_samples, 2*max_field_size*embedding_length)
    y_new has dimensions (training_samples, number_of_classes)
    """

    x_patchy_nodes, x_patchy_edges, x_embedding, y = format_all_training_data(training_graphs)
    _, counts = np.unique(y, return_counts=True)

    if len(counts) == 1:
        raise ValueError("No training data has been created. Pattern not found.")

    min_count = np.amin(counts)
    x_patchy_nodes, x_patchy_edges, x_embedding, y_train = \
        create_balanced_training_set(x_patchy_nodes, x_patchy_edges, x_embedding, y, min_count)
    print("The training data has been balanced.")

    from keras.utils import to_categorical
    y_new = to_categorical(y_train)

    return shuffle_datasets(x_patchy_nodes, x_patchy_edges, x_embedding, y_new)


def get_final_datasets(results):
    """
    Given a list of BoltStatementResults, each corresponding to training data for one
    training pattern, generates training data and formats it properly.

    :param results: A list of BoltStatementResults
    :return: A tuple of ndarrays (x_patchy_nodes, x_patchy_edges, x_embedding, y_new).
    x_patchy_nodes has dimensions (training_samples, field_count*max_field_size, channel_count)
    x_patchy_edges has dimensions (training_samples, field_count*max_field_size*max_field_size, EDGE_PROP_COUNT)
    x_embedding has dimensions (training_samples, 2*max_field_size*embedding_length)
    y_new has dimensions (training_samples, number_of_classes)
    """

    training_graphs = label_and_process_data(results)
    return process_training_examples(training_graphs)
