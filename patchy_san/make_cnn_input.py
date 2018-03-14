"""
Contains functions to generate input in the form of NumPy arrays for the CNN.
"""

import numpy as np
from patchy_san.parameters import MAX_FIELD_SIZE, STRIDE, FIELD_COUNT, CHANNEL_COUNT, HASH_PROPERTIES
from patchy_san.parameters import HASH_FN, DEFAULT_TENSOR_VAL, MAX_NODES, NODE_TYPE_HASH, VOCAB_SIZE
from patchy_san.parameters import EMBEDDING_LENGTH
from patchy_san.neighborhood_assembly import label_and_order_nodes, get_receptive_field
from patchy_san.graph_normalisation import normalise_receptive_field
from optimisable_functions.hashes import hash_labels_only
from keras.preprocessing.text import hashing_trick
from keras.preprocessing.sequence import pad_sequences
from optimisable_functions.hashes import hash_simhash

TENSOR_UPPER_LIMIT = 7e11
TENSOR_LOWER_LIMIT = 0

EDGE_TYPE_HASH = {
    "GLOB_OBJ_PREV": 1,
    "META_PREV": 2,
    "PROC_OBJ": 4,
    "PROC_OBJ_PREV": 8,
    "PROC_PARENT": 16,
    "COMM": 32,
}

EDGE_STATE_HASH = {
    "RaW": 1,
    "WRITE": 2,
    "READ": 4,
    "NONE": 8,
    "CLIENT": 16,
    "SERVER": 32,
    "BIN": 64,
}


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


def build_groups_of_receptive_fields(graph):
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

    :param graph: A Graph object
    :return: A list of lists of tuples of (list of nodes, list of edges), or a list of lists of tuples of
    receptive fields for nodes and edges.
    Each tuple of lists corresponds to a receptive field, and contains all the nodes and edges in it.
    Each list of tuples of lists corresponds to a group of receptive fields.
    The list of lists of tuples of lists corresponds to all the groups of receptive fields found.
    """

    nodes_list = label_and_order_nodes(graph)
    groups_of_receptive_fields = []
    receptive_field = []
    nodes_iter = iter(nodes_list)
    root_node = next(nodes_iter, None)
    receptive_field_count = 0

    while root_node is not None:
        receptive_field_graph = get_receptive_field(root_node.id, graph)
        r_field_nodes_list = normalise_receptive_field(receptive_field_graph)
        edges_list = get_related_edges(r_field_nodes_list, graph)

        receptive_field.append((r_field_nodes_list, edges_list))
        root_node = iterate(nodes_iter, STRIDE)
        receptive_field_count += 1

        if receptive_field_count == FIELD_COUNT:
            groups_of_receptive_fields.append(receptive_field)
            receptive_field = []
            receptive_field_count = 0
    # only whole groups? or partial groups also
    # if norm_fields_list:
    #     groups_of_receptive_fields.append(norm_fields_list)

    return groups_of_receptive_fields


def get_related_edges(nodes_list, graph):
    """
    Returns all edges between nodes in a given list. All the nodes and edges are part of a graph
    which is given as a Graph object.

    :param nodes_list: A list of nodes
    :param graph: A Graph object
    :return: A list of edges
    """

    node_id_list = map(lambda x: x.id, nodes_list)
    node_id_set = set(node_id_list)
    edges = []

    for node in nodes_list:
        if node.id in graph.incoming_edges:
            for edge in graph.incoming_edges[node.id]:

                if edge.start in node_id_set:
                    edges.append(edge)

    return edges


def normalise_tensor(tensor):
    """
    Normalises the tensor by applying computing the following for every element val in
    the tensor:

    new_val = (val-minimum_value)/(maximum_value-minimum_value)

    where minimum_value and maximum_value are the minimum and maximum values in the tensor.

    :param tensor: A 3d NumPy array
    :return: A 3d NumPy array
    """

    normalised_tensor = (tensor-TENSOR_LOWER_LIMIT)/(TENSOR_UPPER_LIMIT-TENSOR_LOWER_LIMIT)
    return normalised_tensor


def build_tensor_naive_hashing(norm_fields_list):
    """
    From a list of receptive fields(list of lists of nodes), builds a 3d NumPy array, with
    the extra dimension coming from the properties extracted from the nodes. This function
    naively applies the same hash function to every string property, and is intended to just
    be a way to help me get the CNN pipeline running.

    :param norm_fields_list: The list of tuples of (list of nodes, list of edges) containing
    the receptive fields.
    :return: A 3d NumPy array
    """

    field_count = len(norm_fields_list)
    tensor = np.zeros((field_count, MAX_FIELD_SIZE, CHANNEL_COUNT), dtype='int64')

    # fields_idx iterates over the receptive fields
    for fields_idx in range(field_count):
        # The first element in the tuple is the list of nodes
        field = norm_fields_list[fields_idx][0]
        # field_idx iterates over the nodes in a receptive field
        for field_idx in range(len(field)):
            node = field[field_idx]
            node_prop = node.properties
            for property_idx in range(CHANNEL_COUNT):
                prop = HASH_PROPERTIES[property_idx]
                if prop in node_prop and node_prop[prop] != []:
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

    return normalise_tensor(tensor)


def build_embedding(graph):
    """
    Given a graph, creates word embeddings for the names of all nodes.

    First the nodes in the graph are ordered by their labels. Next, for each
    node in the ordered list of nodes, a one hot encoding is computed for its name,
    which is a list of integers. This list is padded. The padded list of integers for all nodes
    are combines to form a single list of integers, which is returned.

    :param graph: A Graph object describing the input data
    :return: A 1D numpy array of shape (MAX_NODES*EMBEDDING_LENGTH,)
    """

    nodes_list = list(graph.nodes.values())
    sorted_nodes = sorted(
        nodes_list,
        key=lambda x: hash_labels_only(labels=x.labels, node_label_hash=NODE_TYPE_HASH))
    embedding = []

    for i in range(MAX_NODES):
        if i < len(sorted_nodes) and "name" in sorted_nodes[i].properties and \
                        sorted_nodes[i].properties["name"] != []:

            # The 'name' property on each node is a list, the current solution is to
            # take the first element.
            name = sorted_nodes[i].properties["name"][0]
            encoded_name = hashing_trick(name, VOCAB_SIZE, hash_simhash)

            if "cmdline" in sorted_nodes[i].properties:
                cmdline = sorted_nodes[i].properties["cmdline"]
                encoded_cmdline = hashing_trick(cmdline, VOCAB_SIZE, hash_simhash)
            else:
                encoded_cmdline = []

            embedding += [encoded_name, encoded_cmdline]
        else:
            embedding += [[], []]

    padded_embedding = pad_sequences(embedding, maxlen=EMBEDDING_LENGTH)
    combined_embedding = [num for sublist in padded_embedding for num in sublist]
    return np.asarray(combined_embedding, dtype=np.int16)


def build_edges_tensor(norm_fields_list):
    """
    Given a list of tuples of (list of nodes, list of edges), builds the input tensor for the
    edges.

    We iterate over all the receptive fields. For each receptive field, we build an adjacency
    matrix corresponding to the edges between nodes in the receptive field. Since the list of nodes
    given is normalised already, we can use this ordering for the adjacency matrix.

    For each edge in a receptive field, we assign a number for the edge label and the edge state.
    We thus produce a tensor with dimensions (FIELD_COUNT, MAX_NODES, MAX_NODES, 2).
    We then reshape it to a 2D tensor: (FIELD_COUNT*MAX_NODES*MAX_NODES, 2)

    :param norm_fields_list: A list of tuples of (list of nodes, list of edges).
    Each tuple describes a receptive field, and contains all the nodes and edges in this field.
    We may have multiple receptive fields, hence we have a list of tuples.
    :return: A NumPy ndarray with dimensions (FIELD_COUNT*MAX_NODES*MAX_NODES, 2)
    """

    tensor = np.zeros((FIELD_COUNT, MAX_NODES, MAX_NODES, 2), dtype='int64')

    # fields_idx iterates over the receptive fields
    for fields_idx in range(FIELD_COUNT):
        # The normalised list of nodes is the first item in the tuple
        recept_field_nodes = norm_fields_list[fields_idx][0]

        # The normalised list of edges is the second item in the tuple
        recept_field_edges = norm_fields_list[fields_idx][1]

        node_id_to_position = {}

        # Record the position of each node in the adjacency matrix
        for idx in range(len(recept_field_nodes)):
            node_id = recept_field_nodes[idx].id
            node_id_to_position[node_id] = idx

        for edge in recept_field_edges:
            start_pos = node_id_to_position[edge.start]
            end_pos = node_id_to_position[edge.end]
            tensor[fields_idx][start_pos][end_pos][0] = EDGE_TYPE_HASH[edge.type]
            tensor[fields_idx][start_pos][end_pos][1] = EDGE_STATE_HASH[edge.properties["state"]]

    tensor.reshape((FIELD_COUNT*MAX_NODES*MAX_NODES, 2))
