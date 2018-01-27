"""
Contains functions to return lists of subgraphs which matches certain patterns.
(or does not match any pattern)
"""

from data_processing.preprocessing import build_in_out_edges, get_nodes_edges_by_result
from make_training_data.training_pattern_checks import matches_download_file_write
from make_training_data.fetch_training_data import get_train_all_triples

import make_training_data.training_pattern_checks as checks

# TODO: reuse code by creating generic function to iterate over nodes in each subgraph. D-R-Y


def get_training_data():
    """
    Gets instances of a node connecting to a socket and writing to a file. The input
    BoltStatementResult should describe subgraphs with 3 nodes at maximum.

    :return: A list of tuples of (label, nodes, edges). label is an integer,
    nodes is a Dictionary of node_id -> node, edges is a Dictionary of edge_id -> edge
    """

    results = get_train_all_triples()
    training_data = []
    nodes_edges_list = get_nodes_edges_by_result(results)

    for training_nodes, training_edges in nodes_edges_list:
        incoming_edges, _ = build_in_out_edges(training_edges)
        has_match = False
        for node_id in training_nodes:
            if matches_download_file_write(node_id, training_nodes, incoming_edges):
                training_data.append((1, training_nodes, training_edges))
                has_match = True
                break
        if not has_match:
            training_data.append((0, training_nodes, training_edges))

    return training_data
