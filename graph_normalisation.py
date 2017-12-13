"""
Contains functions to normalise graph nodes in a linear ordering such that similar graphs
have nodes ordered similarly (relative ordering of nodes) after being normalised.
"""

from preprocessing import *

HASH_PROPERTIES = ['cmdline', 'name', 'ips', 'client_port', 'meta_login']
NODE_TYPE_HASH = {'Conn': 2, 'File': 4, 'Global': 8, 'Machine': 16, 'Meta': 32, 'Process': 64,
                  'Socket': 128}


def build_normalised_adj_matrix(results):
    """
    Builds a normalised adjacency matrix using hashing of several properties for ordering.

    :param results: A BoltStatementResult matrix
    :return:
    """

    nodes, edges = get_nodes_edges(results)
    incoming_edges, outgoing_edges = build_in_out_edges(edges)
    consolidate_node_versions(nodes, edges, incoming_edges, outgoing_edges)
    remove_anomalous_nodes_edges(nodes, edges, incoming_edges, outgoing_edges)

    node_hash_list = []

    for node_id in nodes.keys():
        node = nodes[node_id]
        node_hash = compute_hash(node, hash)
        node_hash_list.append((node_hash, node))

    node_hash_list = sorted(node_hash_list, key=lambda x: x[0])
    node_count = len(nodes)
    adjacency_matrix = np.matrix(np.zeros(shape=(node_count, node_count), dtype=np.int8))
    id_to_index = {}

    idx = 0
    for _, node in node_hash_list:
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
            hash_value += abs(int(str(prop_hash)[:4]))

    return hash_value
