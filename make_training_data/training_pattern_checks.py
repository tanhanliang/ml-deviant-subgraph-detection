"""
Contains functions to check if subgraphs match certain patterns, given a start node id.
"""


def matches_download_file_write(node_id, nodes, incoming_edges):
    """
    Checks if a node is the common successor of a subgraph of at least 3 nodes in the following
    configuration: (m1:File)-[r1.state = Write or r1.state = RaW]->(node)<-[]-(m2:Socket).

    :param node_id: The id of the node to check
    :param nodes: A Dictionary of node_id -> node
    :param incoming_edges: A Dictionary of node_id -> list of edges
    :return: A boolean value
    """
    if node_id not in incoming_edges:
        return False
    node_incoming_edges = incoming_edges[node_id]
    found_socket = False
    found_file = False

    for edge in node_incoming_edges:
        predecessor_node = nodes[edge.start]
        edge_prop = edge.properties
        if "Socket" in predecessor_node.labels:
            found_socket = True

        elif "state" in edge_prop and \
            (edge_prop["state"] == "RaW" or edge_prop["state"] == "WRITE") and \
                "File" in predecessor_node.labels:
            found_file = True

    return found_file and found_socket
