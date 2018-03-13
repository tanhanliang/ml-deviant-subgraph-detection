"""
Contains functions to normalise graph nodes in a linear ordering such that similar graphs
have nodes ordered similarly (relative ordering of nodes) after being normalised.
"""

from patchy_san.parameters import HASH_PROPERTIES, NODE_TYPE_HASH, PROPERTY_CARDINALITY, RECEPTIVE_FIELD_HASH

EDGE_TYPE_HASH = {
    "GLOB_OBJ_PREV": 1,
    "META_PREV": 2,
    "PROC_OBJ": 4,
    "PROC_OBJ_PREV": 8,
    "PROC_PARENT": 16,
    "COMM": 32,
}

EDGE_STATE_HASH = {
    "RaW": 64,
    "WRITE": 128,
    "READ": 256,
    "NONE": 512,
    "CLIENT": 1024,
    "SERVER": 2048,
    "BIN": 4096,
}


def normalise_receptive_field(graph):
    """
    Builds a list of nodes and orders them in ascending order using the hash function
    provided.

    :param graph: A Graph object
    :return: A list of nodes ordered using the hash fn.
    """

    node_list = list(graph.nodes.values())
    return sorted(node_list, key=lambda node: compute_hash(node))


def normalise_edge_list(edges_list):
    """
    Sorts a list of edges using a number computed from the edge label and edge state.

    :param edges_list: A list of edges
    :return: A list of edges
    """

    return sorted(edges_list, key=lambda edge: compute_edge_hash(edge))


def compute_edge_hash(edge):
    """
    Computes a value for an edge.

    :param edge: An edge
    :return: An integer
    """

    if "state" in edge.properties:
        val = EDGE_STATE_HASH[edge.properties["state"]]

    else:
        val = 0

    return val + EDGE_TYPE_HASH[edge.type]


def compute_hash(node):
    """
    Given a Node, computes a hash value based on a given hash function,
    the Node type and several properties.

    :param node: A neo4j Node
    :return: A hash value as a long integer
    """
    hash_value = 0
    for label in node.labels:
        hash_value += NODE_TYPE_HASH[label]

    properties = node.properties

    for prop in HASH_PROPERTIES:
        hash_value *= PROPERTY_CARDINALITY[prop]
        if properties.__contains__(prop) and properties[prop] != []:
            if prop == 'name':
                # A node may have multiple names, use only the first
                prop_hash = RECEPTIVE_FIELD_HASH(properties[prop][0])
            else:
                prop_hash = RECEPTIVE_FIELD_HASH(properties[prop])
            # Take the 4 most significant digits
            hash_value += int(str(abs(prop_hash))[:PROPERTY_CARDINALITY[prop]])

    return hash_value
