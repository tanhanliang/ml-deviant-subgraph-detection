"""
This module contains functions to process graph data into a form usable by the Patchy-San
algorithm.
"""
import queue
from patchy_san.parameters import MAX_FIELD_SIZE as SIZE


def label_and_order_nodes(nodes, transform_fn=None):
    """
    Sorts a list of nodes by some labeling function, for example sorting by node timestamp.

    :param transform_fn: A function which takes a node as an argument and returns some value
    to be used for sorting
    :param nodes: A Dictionary of node_id -> node
    :return: A list of sorted nodes
    """
    if transform_fn is None:
        from patchy_san.parameters import LABELING_FN
        transform_fn = LABELING_FN

    nodes_list = list(nodes.values())
    nodes_list = sorted(nodes_list, key=transform_fn)
    return nodes_list


def get_receptive_field(root_id, nodes, incoming_edges):
    """
    Given a root node, performs breadth-first search, adding explored nodes to a Set.
    If number of reachable nodes is less than size, no padding is done.
    Selects next node in the breadth-first search arbitrarily (among all nodes with
    same distance from root).

    We explore nodes in the opposite direction to the edges, because the edge directions
    represent the happens-after relation, so (n1) -> (n2) means that (n2) came first and
    spawned (n1).

    TODO: Find way to implement ordering such that nodes will not be selected arbitrarily.

    :param root_id: The id of the start node
    :param nodes: A Dictionary of node_id -> node
    :param incoming_edges: A Dictionary of node_id -> list of outgoing edges
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

    while neighborhood_size < SIZE:
        if node_edge_q.empty():
            # No padding if size of graph smaller than desired receptive field
            break
        else:
            item = node_edge_q.get(block=False)
            node = item[0]
            edge = item[1]
            nodes_dict[node.id] = node
            # All edges except the root will have a predecessor
            if edge is not None:
                edges_dict[edge.id] = edge

            neighborhood_size += 1

            if node.id in incoming_edges.keys():
                for edge in incoming_edges[node.id]:
                    neighbor_node = nodes[edge.start]
                    if neighbor_node not in marked_set:
                        node_edge_q.put((neighbor_node, edge))
                        marked_set.add(neighbor_node)

    return nodes_dict, edges_dict
