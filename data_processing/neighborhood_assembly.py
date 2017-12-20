"""
This module contains functions to process graph data into a form usable by the Patchy-San
algorithm.
"""
import queue


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
