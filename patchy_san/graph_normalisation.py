"""
Contains functions to normalise graph nodes in a linear ordering such that similar graphs
have nodes ordered similarly (relative ordering of nodes) after being normalised.
"""

from patchy_san.parameters import HASH_PROPERTIES, NODE_TYPE_HASH, PROPERTY_CARDINALITY, RECEPTIVE_FIELD_HASH


def normalise_receptive_field(graph):
    """
    Builds a list of nodes and orders them in ascending order using the hash function
    provided.

    :param graph: A Graph object
    :return: A list of nodes ordered using the hash fn.
    """

    node_list = list(graph.nodes.values())
    return sorted(node_list, key=lambda node: compute_hash(node))


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
                prop_hash = RECEPTIVE_FIELD_HASH(property=properties[prop][0])
            else:
                prop_hash = RECEPTIVE_FIELD_HASH(property=properties[prop])
            # Take the 4 most significant digits
            hash_value += int(str(abs(prop_hash))[:PROPERTY_CARDINALITY[prop]])

    return hash_value
