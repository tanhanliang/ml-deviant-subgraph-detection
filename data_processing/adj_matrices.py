"""
Contains a representation of an adjacency matrix as a class
"""

class AdjacencyMatrix:
    """Wrapper class to represent an Adjacency Matrix"""
    def __init__(self, matrix, id_to_index, nodes, edges):
        """
        Initialises wrapper object with all the data it needs to usefully
        represent a graph.

        :param matrix: Adjacency matrix as a NumPy matrix. All edge weights are '1'.
        :param id_to_index: Dictionary of Node id -> row/column index in matrix.
        :param nodes: Dictionary of Node id -> Node
        :param edges: Dictionary of Edge id -> Edge
        """

        self.matrix = matrix
        self.id_to_index = id_to_index
        self.nodes = nodes
        self.edges = edges
