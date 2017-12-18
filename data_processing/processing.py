"""
This module contains functions to process graph data into a form usable by the Patchy-San
algorithm.
"""


def generate_node_list(node_cmp_fn, nodes):
    """
    Given a function used to order nodes and a Dictionary of node_id -> node which represents all
    nodes in the graph, returns a list of nodes ordered by the labeling function.

    The node ordering function compares two nodes and returns:
    (1) -1 if the first node is ordered before the second node,
    (2) 0 if the nodes are equivalent in order, and
    (3) 1 if the first node is ordered after the second node.

    :param node_cmp_fn: A function which takes two nodes as arguments and returns an integer
    :param nodes: A Dictionary of node_id -> node
    :return: A list of nodes in the correct order
    """

    nodes_list = list(nodes.values())
    sorted(nodes_list, cmp=node_cmp_fn)
    return nodes_list


def ts_ordering_asc(node1, node2):
    """
    Generates an integer representing the relative ordering of the nodes based on timestamp.
    If node1's timestamp is smaller than node2's timestamp, then node1 is ordered before node2,
    and vice versa. This function assumes that a timestamp exists in the properties dict of the
    node, otherwise it will throw an exception.

    :param node1: First node to compare
    :param node2: Second node to compare
    :return: An integer representing the relative ordering of node1 compared to node2
    """

    if 'timestamp' not in node1.properties or 'timestamp' not in node2.properties:
        raise RuntimeError('timestamp does not exist in properties dict of node')
    node1_ts = node1.properties['timestamp']
    node2_ts = node2.properties['timestamp']

    if node1_ts < node2_ts:
        return -1
    elif node1_ts == node2_ts:
        return 0
    else:
        return 1
