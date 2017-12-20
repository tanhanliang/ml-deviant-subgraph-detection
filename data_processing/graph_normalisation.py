"""
Contains functions to normalise graph nodes in a linear ordering such that similar graphs
have nodes ordered similarly (relative ordering of nodes) after being normalised.
"""

from data_processing.preprocessing import *

HASH_PROPERTIES = ['cmdline', 'name', 'ips', 'client_port', 'meta_login']
NODE_TYPE_HASH = {'Conn': 2, 'File': 4, 'Global': 8, 'Machine': 16, 'Meta': 32, 'Process': 64,
                  'Socket': 128}


def build_node_list_hashing(nodes, hash_fn):
    """
    Builds a list of nodes and orders them in ascending order using the hash function
    provided.

    :param nodes: A Dictionary of node_id -> node
    :param hash_fn: The hash function to use. E.g. SimHash, MD5
    :return: A list of nodes ordered using the hash fn.
    """

    node_list = list(nodes.values())
    return sorted(node_list, key=lambda node: compute_hash(node, hash_fn))


def build_normalised_adj_matrix(results, hash_fn):
    """
    Builds a normalised adjacency matrix using hashing of several properties for ordering.

    :param results: A BoltStatementResult matrix
    :param hash_fn: The hash function to use
    :return:An adjacency matrix representation for the graph, using a np.matrix
    """

    nodes, edges = get_nodes_edges(results)
    incoming_edges, outgoing_edges = build_in_out_edges(edges)
    consolidate_node_versions(nodes, edges, incoming_edges, outgoing_edges)
    remove_anomalous_nodes_edges(nodes, edges, incoming_edges, outgoing_edges)

    ordered_node_list = build_node_list_hashing(nodes, hash_fn)

    node_count = len(nodes)
    adjacency_matrix = np.matrix(np.zeros(shape=(node_count, node_count), dtype=np.int8))
    id_to_index = {}

    idx = 0
    for _, node in ordered_node_list:
        id_to_index[node.id] = idx
        idx += 1

    for edge_id in edges.keys():
        start_idx = id_to_index[edges[edge_id].start]
        end_idx = id_to_index[edges[edge_id].end]
        adjacency_matrix[start_idx, end_idx] = 1

    return AdjacencyMatrix(adjacency_matrix, id_to_index, nodes, edges)


def compute_hash(node, hash_fn):
    """
    Given a Node, computes a hash value based on a given hash function,
    the Node type and several properties.

    :param node: A neo4j Node
    :param hash_fn: A given hash function
    :return: A hash value as a long integer
    """
    hash_value = 0
    for label in node.labels:
        hash_value += NODE_TYPE_HASH[label]

    properties = node.properties

    for prop in HASH_PROPERTIES:
        hash_value *= 10000
        if properties.__contains__(prop):
            if prop == 'name':
                # A node may have multiple names, use only the first
                # TODO: Update with better solution after meeting with supervisor
                prop_hash = hash_fn(properties[prop][0])
            else:
                prop_hash = hash_fn(properties[prop])
            # Take the 4 most significant digits
            hash_value += int(str(abs(prop_hash))[:4])

    return hash_value
