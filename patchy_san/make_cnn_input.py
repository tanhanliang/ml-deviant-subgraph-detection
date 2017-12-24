"""
Contains functions to generate input in the form of NumPy arrays for the CNN.
"""

import numpy as np
from patchy_san.neighborhood_assembly import generate_node_list, get_receptive_field
from data_processing.preprocessing import build_in_out_edges


def iterate(iterator, n):
    """
    Returns the nth element returned by the iterator if there are sufficient elements,
    or None if the iterator has been exhausted.

    :param iterator: The iterator to extract elements from
    :param n: The nth element returned by the iterator will be returned
    :return: An element returned by the iterator
    """
    # throw away n-1 elements
    for index in range(1, n):
        next(iterator, None)

    return next(iterator, None)


def build_node_receptive_fields(nodes, edges, labeling_fn, norm_field_fn, field_size, stride):
    """
    Builds the receptive fields for nodes in a given graph. First the nodes
    are ordered by a given labeling function. Starting with the first node
    in this list, receptive fields are built using nodes in the ordered list as root nodes,
    with distance stride between root nodes in the ordered list (e.g if stride is 1 then
    all nodes in the ordered list will be root nodes). The receptive fields are built in the form
    of lists, which are built (and ordered) using a given function.

    :param nodes: A Dictionary of node_id -> node
    :param edges: A Dictionary of edge_id -> edge
    :param labeling_fn: A function used to transform the node list before sorting, using
    the built in sorting fn. E.g sorted(node_list, key=labeling_fn)
    :param norm_field_fn: A function used to build the normalised node list for each field.
    This function should only take a Dictionary of node_id -> node as input
    :param field_size: Size of the receptive field
    :param stride: Distance between chosen root nodes after graph has been transformed
    into list of nodes
    :return: A list of lists of nodes, or a list of receptive fields
    """

    nodes_list = generate_node_list(labeling_fn, nodes)
    norm_fields_list = []
    nodes_iter = iter(nodes_list)
    root_node = next(nodes_iter, None)
    incoming_edges, outgoing_edges = build_in_out_edges(edges)

    while root_node is not None:
        r_field_nodes, r_field_edges = get_receptive_field(root_node.id, nodes, outgoing_edges, field_size)
        r_field_nodes_list = norm_field_fn(r_field_nodes)
        norm_fields_list.append(r_field_nodes_list)
        root_node = iterate(nodes_iter, stride)

    return norm_fields_list
