"""
Contains functions to return lists of subgraphs which matches certain patterns.
(or does not match any pattern)
"""

from data_processing.preprocessing import get_nodes_edges_by_result
from make_training_data.fetch_training_data import get_train_download_file_execute, get_negative_data


def get_training_data():
    """
    Gets instances of a node connecting to a socket and writing to a file. The input
    BoltStatementResult should describe subgraphs with 3 nodes at maximum.

    :return: A list of tuples of (label, nodes, edges). label is an integer,
    nodes is a Dictionary of node_id -> node, edges is a Dictionary of edge_id -> edge
    """

    results = []
    training_data = []
    results.append(get_train_download_file_execute())
    results.append(get_negative_data())
    label = 0

    for result in results:
        nodes_edges_list = get_nodes_edges_by_result(result)

        for training_nodes, training_edges in nodes_edges_list:
            training_data.append((label, training_nodes, training_edges))

        label += 1

    return training_data
