"""
This module contains functions to process graph data into a form usable by the Patchy-San
algorithm.
"""
import queue


def generate_node_list(transform_fn, nodes):
    """
    Sorts a list of nodes by some labeling function, for example sorting by node timestamp.

    :param transform_fn: A function which takes a node as an argument and returns some value
    to be used for sorting
    :param nodes: A Dictionary of node_id -> node
    :return: A list of sorted nodes
    """

    nodes_list = list(nodes.values())
    nodes_list = sorted(nodes_list, key=transform_fn)
    return nodes_list


def get_ts(node):
    """
    Given a node, returns its timestamp if it exists, otherwise throws a RuntimeError.
    This fn will be used to sort a list of nodes by timestamp, using the built in sorted()
    function.

    :param node: A node in a list to be sorted
    :return: The timestamp of the node
    """

    if 'timestamp' not in node.properties:
        raise RuntimeError('timestamp does not exist in properties dict of node')

    return node.properties['timestamp']


def get_receptive_field(root_id, nodes, outgoing_edges, size):
    """
    Given a root node, performs breadth-first search, adding explored nodes to a Set.
    If number of reachable nodes is less than size, no padding is done.
    Selects next node in the breadth-first search arbitrarily (among all nodes with
    same distance from root).
    TODO: Find way to implement ordering such that nodes will not be selected arbitrarily.

    :param root_id: The id of the start node
    :param nodes: A Dictionary of node_id -> node
    :param outgoing_edges: A Dictionary of node_id -> list of outgoing edges
    :param size: The size of the neighborhood.
    :return: A tuple of (node_id -> node, edge_id -> edge) which represents the
    receptive field (which is a subgraph)
    """

    nodes_dict = {}
    edges_dict = {}
    marked_set = set()
    neighborhood_size = 0
    # A queue that contains a tuple of (node, edge) where edge is the edge to the previous
    # explored node
    node_edge_q = queue.Queue()
    node_edge_q.put((nodes[root_id], None))
    marked_set.add(nodes[root_id])

    while neighborhood_size < size:
        if node_edge_q.empty():
            # No padding if size of graph smaller than desired receptive field
            break
        else:
            item = node_edge_q.get(block=False)
            node = item[0]
            edge = item[1]
            nodes_dict[node.id] = node
            # Root node has no edge to predecessor
            if edge is not None:
                edges_dict[edge.id] = edge

            neighborhood_size += 1

            if node.id in outgoing_edges.keys():
                for edge in outgoing_edges[node.id]:
                    neighbor_node = nodes[edge.end]
                    if neighbor_node not in marked_set:
                        node_edge_q.put((neighbor_node, edge))
                        marked_set.add(neighbor_node)

    return nodes_dict, edges_dict
