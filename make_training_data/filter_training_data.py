"""
Contains functions to return lists of subgraphs which matches certain patterns.
(or does not match any pattern)
"""

from data_processing.preprocessing import clean_data, build_in_out_edges
from make_training_data.training_pattern_checks import matches_download_file_write
from make_training_data.fetch_training_data import get_all_triples


def get_download_file_write():
    """
    Gets instances of a node connecting to a socket and writing to a file. The input
    BoltStatementResult should describe subgraphs with 3 nodes at maximum.

    :return: A list of tuples of (nodes, edges). nodes is a Dictionary of node_id -> node, edges
    is a Dictionary of edge_id -> edge
    """

    results = get_all_triples()
    training_data = []
    # Each subgraph_dict in results.data() corresponds to one subgraph
    for subgraph_dict in results.data():
        training_nodes = {}
        training_edges = {}
        for path in subgraph_dict:
            for node in subgraph_dict[path].nodes:
                training_nodes[node.id] = node
            for edge in subgraph_dict[path].relationships:
                training_edges[edge.id] = edge

        clean_data(training_nodes, training_edges)
        incoming_edges, _ = build_in_out_edges(training_edges)

        for node_id in training_nodes:
            if matches_download_file_write(node_id, training_nodes, incoming_edges):
                training_data.append((training_nodes, training_edges))
                break

    return training_data
