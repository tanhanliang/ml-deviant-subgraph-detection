"""
Contains functions to return lists of subgraphs which matches certain patterns.
(or does not match any pattern)
"""

from data_processing.preprocessing import build_in_out_edges, get_nodes_edges_by_result
from make_training_data.training_pattern_checks import matches_download_file_write
from make_training_data.fetch_training_data import get_all_triples

import make_training_data.training_pattern_checks as checks

# TODO: reuse code by creating generic function to iterate over nodes in each subgraph


def get_download_file_write():
    """
    Gets instances of a node connecting to a socket and writing to a file. The input
    BoltStatementResult should describe subgraphs with 3 nodes at maximum.

    :return: A list of tuples of (nodes, edges). nodes is a Dictionary of node_id -> node, edges
    is a Dictionary of edge_id -> edge
    """

    results = get_all_triples()
    training_data = []
    nodes_edges_list = get_nodes_edges_by_result(results)

    for training_nodes, training_edges in nodes_edges_list:
        incoming_edges, _ = build_in_out_edges(training_edges)
        for node_id in training_nodes:
            if matches_download_file_write(node_id, training_nodes, incoming_edges):
                training_data.append((training_nodes, training_edges))
                break

    return training_data


def get_negative_data():
    """
    Gets data which does not match any pattern.

    :return: A list of tuples of (nodes, edges). nodes is a Dictionary of node_id -> node, edges
    is a Dictionary of edge_id -> edge
    """
    import time
    # start = time.time()
    results = get_all_triples()
    training_data = []
    nodes_edges_list = get_nodes_edges_by_result(results)

    for training_nodes, training_edges in nodes_edges_list:
        has_match = False
        incoming_edges, _ = build_in_out_edges(training_edges)
        for node_id in training_nodes:
            if has_match:
                break
            for item in dir(checks):
                attribute = getattr(checks, item)
                if callable(attribute) and item.startswith('matches'):
                    if attribute(node_id, training_nodes, incoming_edges):
                        has_match = True
                        break
        if not has_match:
            training_data.append((training_nodes, training_edges))
    # end = time.time()
    # print("Time taken to process " + str(len(nodes_edges_list)) + " subgraphs:")
    # print("In seconds: " + str(end-start))
    return training_data
