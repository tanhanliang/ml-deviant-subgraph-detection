"""
Contains functions to generate input in the form of NumPy arrays for the CNN.
"""

import numpy as np
from patchy_san.neighborhood_assembly import get_receptive_field
from data_processing.preprocessing import build_in_out_edges
from patchy_san.parameters import MAX_FIELD_SIZE, STRIDE, FIELD_COUNT, CHANNEL_COUNT, HASH_PROPERTIES
from patchy_san.parameters import HASH_FN, DEFAULT_TENSOR_VAL
from patchy_san.graph_normalisation import NODE_TYPE_HASH
from optimisable_functions.hashes import hash_labels_prop


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


def build_groups_of_receptive_fields(nodes, edges, norm_field_fn=None):
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
    :param norm_field_fn: A function used to build the normalised node list for each field.
    This function should only take a Dictionary of node_id -> node as input
    into list of nodes e.g:
    norm_field_fn = build_node_list_hashing
    :return: A list of lists of lists of nodes, or a list of lists of receptive fields
    """

    if norm_field_fn is None:
        from patchy_san.neighborhood_assembly import generate_node_list
        norm_field_fn = generate_node_list

    nodes_list = norm_field_fn(nodes)
    groups_of_receptive_fields = []
    norm_fields_list = []
    nodes_iter = iter(nodes_list)
    root_node = next(nodes_iter, None)
    incoming_edges, outgoing_edges = build_in_out_edges(edges)
    norm_fields_count = 0

    while root_node is not None:
        r_field_nodes, r_field_edges = get_receptive_field(root_node.id, nodes, incoming_edges)
        r_field_nodes_list = norm_field_fn(r_field_nodes)
        norm_fields_list.append(r_field_nodes_list)
        root_node = iterate(nodes_iter, STRIDE)
        norm_fields_count += 1

        if norm_fields_count == FIELD_COUNT:
            groups_of_receptive_fields.append(norm_fields_list)
            norm_fields_list = []
            norm_fields_count = 0
    # only whole groups? or partial groups also
    # if norm_fields_list:
    #     groups_of_receptive_fields.append(norm_fields_list)

    return groups_of_receptive_fields


def build_tensor_naive_hashing(norm_fields_list):
    """
    From a list of receptive fields(list of lists of nodes), builds a 3d NumPy array, with
    the extra dimension coming from the properties extracted from the nodes. This function
    naively applies the same hash function to every string property, and is intended to just
    be a way to help me get the CNN pipeline running.

    This method should be replaced in the near future.

    :param norm_fields_list: The list of lists of nodes containing the receptive fields
    :return: A 3d NumPy array
    """

    field_count = len(norm_fields_list)
    tensor = np.zeros((field_count, MAX_FIELD_SIZE, CHANNEL_COUNT), dtype='int64')

    for fields_idx in range(field_count):
        field = norm_fields_list[fields_idx]
        for field_idx in range(len(field)):
            node = field[field_idx]
            node_prop = node.properties
            for property_idx in range(CHANNEL_COUNT):
                prop = HASH_PROPERTIES[property_idx]
                if prop in node_prop and node_prop[prop] != []:
                    # TODO: Ask supervisor about better way to do this
                    if prop == 'name':
                        val = HASH_FN(
                            labels=node.labels,
                            node_label_hash=NODE_TYPE_HASH,
                            property=node_prop[prop][0])
                    else:
                        val = HASH_FN(
                            labels=node.labels,
                            node_label_hash=NODE_TYPE_HASH,
                            property=node_prop[prop])
                else:
                    val = DEFAULT_TENSOR_VAL
                tensor[fields_idx][field_idx][property_idx] = val

    return tensor
