"""
Contains graph representation(s).
"""


class Graph:
    """
    A Graph object which encapsulates all data needed to represent it.
    """

    def __init__(self, nodes, edges, incoming_edges, outgoing_edges):
        """
        Initialises the Graph object.

        :param nodes: A Dictionary of node_id to Neo4j Nodes
        :param edges: A Dictionary of edge_id to edges
        :param incoming_edges: A Dictionary of node_id -> list of edges (incoming edges to that node)
        :param outgoing_edges: A Dictionary of node_id -> list of edges (outgoing edges from that node)
        """

        self.nodes = nodes
        self.edges = edges
        self.incoming_edges = incoming_edges
        self.outgoing_edges = outgoing_edges
