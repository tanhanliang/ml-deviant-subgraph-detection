"""
Contains functions to format the training data into ndarrays that can be used to train the
model.
"""
from patchy_san.make_cnn_input import build_groups_of_receptive_fields, build_tensor_naive_hashing
from patchy_san.graph_normalisation import HASH_PROPERTIES
import make_training_data.fetch_training_data as fetch_training_data
from data_processing.preprocessing import clean_data
import numpy as np


def get_all_training_data(labeling_fn, norm_field_fn, field_count, max_field_size, stride):
    """
    Queries the database for data according to several predefined rules, then processes
    them into two ndarrays.

    :param labeling_fn: A function used to transform the node list before sorting, using
    the built in sorting fn. E.g sorted(node_list, key=labeling_fn)
    :param norm_field_fn: A function used to build the normalised node list for each field.
    This function should only take a Dictionary of node_id -> node as input
    :param field_count: Max number of receptive fields
    :param max_field_size: Max receptive field size
    :param stride: Distance between chosen root nodes after graph has been transformed
    into list of nodes
    :return: A tuple (x_data,y_target) of ndarrays. x_data has shape (s,field_count,max_field_size,n)
    and y_target has shape (s,1)
    s: number of training samples
    n: number of attributes
    """
    x_data_list = []
    y_target_list = []
    target_class = 0

    for item in dir(fetch_training_data):
        attribute = getattr(fetch_training_data, item)
        if callable(attribute) and item.startswith('get'):
            # retrieve the training data as a BoltStatementResult
            results = attribute()
            nodes, edges = clean_data(results)
            receptive_fields_groups = build_groups_of_receptive_fields(
                nodes, edges, labeling_fn, norm_field_fn, field_count, max_field_size, stride
            )
            for fields_list in receptive_fields_groups:
                training_example_tensor = build_tensor_naive_hashing(fields_list, hash, max_field_size)
                x_data_list.append(training_example_tensor)
                y_target_list.append(target_class)

            target_class += 1

    training_examples = len(x_data_list)
    node_properties = len(HASH_PROPERTIES)
    x_data = np.ndarray((training_examples, field_count, max_field_size, node_properties))
    y_target = np.asarray(y_target_list, dtype=np.int32)

    idx = 0
    while idx < training_examples:
        x_data[idx] = x_data_list[idx]
        idx += 1

    return x_data, y_target
