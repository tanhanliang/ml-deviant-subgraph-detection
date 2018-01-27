"""
Contains functions to format the training data into ndarrays that can be used to train the
model.
"""
from patchy_san.make_cnn_input import build_groups_of_receptive_fields, build_tensor_naive_hashing
import make_training_data.filter_training_data as filter_training_data
from patchy_san.parameters import FIELD_COUNT, MAX_FIELD_SIZE, CHANNEL_COUNT
import numpy as np


def get_all_training_data():
    """
    Queries the database for data according to several predefined rules, then processes
    them into two ndarrays.

    :return: A tuple (x_data,y_target) of ndarrays. x_data has shape (s,field_count,max_field_size,n)
    and y_target has shape (s,1)
    s: number of training samples
    n: number of attributes
    """
    x_data_list = []
    y_target_list = []
    target_class = 0

    for item in dir(filter_training_data):
        attribute = getattr(filter_training_data, item)
        if callable(attribute) and item.startswith('get_filtered_'):
            # training_data is a list of (node_id->node, edge_id->)
            training_data = attribute()

            for (training_nodes, training_edges) in training_data:
                print(training_nodes)
                receptive_fields_groups = build_groups_of_receptive_fields(training_nodes, training_edges)

                # For training data for most classes there should only be one receptive field group
                for fields_list in receptive_fields_groups:
                    training_example_tensor = build_tensor_naive_hashing(fields_list)
                    x_data_list.append(training_example_tensor)
                    y_target_list.append(target_class)

            target_class += 1

    training_examples = len(x_data_list)
    x_data = np.ndarray((training_examples, FIELD_COUNT, MAX_FIELD_SIZE, CHANNEL_COUNT))
    y_target = np.asarray(y_target_list, dtype=np.int32)

    idx = 0
    while idx < training_examples:
        x_data[idx] = x_data_list[idx]
        idx += 1

    return x_data, y_target
