"""
Contains functions to generate input in the form of NumPy arrays for the CNN.
"""

import numpy as np
from patchy_san.neighborhood_assembly import generate_node_list, get_receptive_field
from data_processing.preprocessing import build_in_out_edges
from patchy_san.graph_normalisation import HASH_PROPERTIES


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


def build_groups_of_receptive_fields(nodes, edges, labeling_fn, norm_field_fn, field_count, field_size, stride):
    """
    Extracts as many groups of receptive fields as possible. Each group of fields is considered
    complete once it reaches the maximum field size.

    First the nodes are ordered by a given labeling function. Starting with the first node
    in this list, receptive fields are built using nodes in the ordered list as root nodes,
    with distance stride between root nodes in the ordered list (e.g if stride is 1 then
    all nodes in the ordered list will be root nodes). The receptive fields are built in the form
    of lists, which are built (and ordered) using a given function.

    Each list of receptive fields makes a group. This function returns all groups that could be
    constructed.

    :param nodes: A Dictionary of node_id -> node
    :param edges: A Dictionary of edge_id -> edge
    :param labeling_fn: A function used to transform the node list before sorting, using
    the built in sorting fn. E.g sorted(node_list, key=labeling_fn)
    :param field_count: Max number of receptive fields
    :param norm_field_fn: A function used to build the normalised node list for each field.
    This function should only take a Dictionary of node_id -> node as input
    :param field_size: Size of the receptive field
    :param stride: Distance between chosen root nodes after graph has been transformed
    into list of nodes
    :return: A list of lists of lists of nodes, or a list of lists of receptive fields
    """

    nodes_list = generate_node_list(labeling_fn, nodes)
    groups_of_receptive_fields = []
    norm_fields_list = []
    nodes_iter = iter(nodes_list)
    root_node = next(nodes_iter, None)
    incoming_edges, outgoing_edges = build_in_out_edges(edges)
    norm_fields_count = 0

    while root_node is not None:
        if norm_fields_count == field_count:
            groups_of_receptive_fields.append(norm_fields_list)
            norm_fields_list = []
            norm_fields_count = 0

        r_field_nodes, r_field_edges = get_receptive_field(root_node.id, nodes, incoming_edges, field_size)
        r_field_nodes_list = norm_field_fn(r_field_nodes)
        norm_fields_list.append(r_field_nodes_list)
        root_node = iterate(nodes_iter, stride)
        norm_fields_count += 1

    # only whole groups? or partial groups also
    # if norm_fields_list:
    #     groups_of_receptive_fields.append(norm_fields_list)

    return groups_of_receptive_fields


def build_tensor_naive_hashing(norm_fields_list, hash_fn, max_field_size):
    """
    From a list of receptive fields(list of lists of nodes), builds a 3d NumPy array, with
    the extra dimension coming from the properties extracted from the nodes. This function
    naively applies the same hash function to every string property, and is intended to just
    be a way to help me get the CNN pipeline running.

    This method should be replaced in the near future.

    :param norm_fields_list: The list of lists of nodes containing the receptive fields
    :param hash_fn: The hash function to use
    :param max_field_size: The max receptive field size
    :return: A 3d NumPy array
    """

    field_count = len(norm_fields_list)
    tensor = np.zeros((field_count, max_field_size, len(HASH_PROPERTIES)), dtype='int64')

    for fields_idx in range(field_count):
        field = norm_fields_list[fields_idx]
        for field_idx in range(len(field)):
            node_prop = field[field_idx].properties
            for property_idx in range(len(HASH_PROPERTIES)):
                prop = HASH_PROPERTIES[property_idx]
                if prop in node_prop:
                    # TODO: Ask supervisor about better way to do this
                    if prop == 'name':
                        val = hash_fn(node_prop[prop][0])
                    else:
                        val = hash_fn(node_prop[prop])
                else:
                    val = 0
                tensor[fields_idx][field_idx][property_idx] = val

    return tensor
