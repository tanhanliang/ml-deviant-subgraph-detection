"""
This module contains functions to extract and process data from a neo4j database.

Dependencies: neo4j-driver
"""
from neo4j.v1 import GraphDatabase, basic_auth
import numpy as np

VERSION_TYPES = ['GLOB_OBJ_PREV', 'META_PREV', 'PROC_OBJ_PREV']


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


def get_subgraph_paths(root_id, end_id):
    """
    Queries the neo4j database for all paths starting from a root node and ending
    at an end node

    :param root_id: The database Id of the root node
    :param end_id: The database Id of the end node
    :return: A BoltStatementResult object describing all paths between root node
    specified by root_id, and end node specified by end_id.
    """
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "neo4j"))
    session = driver.session()

    query = """
    MATCH path=(n)-[r*]->(m)
    WHERE Id(n) = $id1 AND Id(m) = $id2
    RETURN path
    """
    results = session.run(query, {"id1": root_id, "id2": end_id})
    session.close()
    return results


def build_adjacency_matrix(results):
    """
    Builds an adjacency matrix based on a given graph, represented as a BoltStatementResult

    :param results: A BoltStatementResult describing all paths within the graph
    :return: An adjacency matrix representation for the graph, using a np.matrix
    """

    nodes = {}
    edges = {}

    for result in results.data():
        for node in result['path'].nodes:
            nodes[node.id] = node
        for edge in result['path'].relationships:
            edges[edge.id] = edge

    nodes, edges = consolidate_node_versions(nodes, edges)

    node_count = len(nodes)
    adjacency_matrix = np.matrix(np.zeros(shape=(node_count, node_count), dtype=np.int8))
    id_to_index = {}

    idx = 0
    for node_id in nodes.keys():
        id_to_index[node_id] = idx
        idx += 1

    for edge_id in edges.keys():
        start_idx = id_to_index[edges[edge_id].start]
        end_idx = id_to_index[edges[edge_id].end]
        adjacency_matrix[start_idx, end_idx] = 1

    return AdjacencyMatrix(adjacency_matrix, id_to_index, nodes, edges)


def consolidate_node_versions(nodes, edges):
    """
    Given a Dictionary of node_id -> node and a Dictionary of edge_id -> edge,
    for all adjacent edges and nodes (node1)-[edge]->(node2) where
    type(edge) is a previous version type, removes edge and node2.
    All outgoing and incoming edges to node2 are glued to node1.

    Note: This may produce a graph where two edges of the same type may exist between
    2 nodes. Currently this is not a problem because the adjacency matrix produced from
    the set of edges and nodes cannot contain duplicated edges.

    :param nodes: A Dictionary of node_id to Neo4j Nodes
    :param edges: A Dictionary of edge_id to edges
    :return: (nodes, edges), a tuple containing a Dictionary of node_id to Neo4j Nodes
    and a Dictionary of edge_id to edges
    """

    incoming_edges, outgoing_edges = build_in_out_edges(nodes, edges)

    # Glue incoming and outgoing edges from the old node to the master node
    for edge_id in list(edges.keys()):
        edge = edges[edge_id]
        if edge.type in VERSION_TYPES:
            # Remove the older node version
            removed_node_id = edge.end
            master_node_id = edge.start

            if removed_node_id in outgoing_edges.keys():
                for outgoing_edge in outgoing_edges[removed_node_id]:
                    outgoing_edge.start = master_node_id
                    # The key master_node_id should definitely exist
                    outgoing_edges[master_node_id] += [outgoing_edge]

            if removed_node_id in incoming_edges.keys():
                for incoming_edge in incoming_edges[removed_node_id]:
                    incoming_edge.end = master_node_id
                    # The key master_node_id may not exist
                    if not incoming_edges.__contains__(master_node_id):
                        incoming_edges[master_node_id] = []
                    incoming_edges[master_node_id] += [incoming_edge]

            nodes.pop(removed_node_id)
            edges.pop(edge_id)

    return nodes, edges


def build_in_out_edges(nodes, edges):
    """
    Given a Dictionary of node_id -> node and a Dictionary of edge_id -> edge, builds two
    dictionaries of node_id -> edge (incoming or outgoing edges from that node).

    :param nodes: A Dictionary of node_id -> node
    :param edges: A Dictionary of edge_id -> edge
    :return: (incoming_edges, outgoing_edges), a tuple of Dictionaries of node_id to
    incoming/outgoing edges to/from that node
    """

    incoming_edges = {}
    outgoing_edges = {}

    # Build maps which store all incoming and outgoing edges for every node
    for edge_id in edges.keys():
        edge = edges[edge_id]
        if not incoming_edges.__contains__(edge.end):
            incoming_edges[edge.end] = []
        incoming_edges[edge.end] += [edge]

        if not outgoing_edges.__contains__(edge.start):
            outgoing_edges[edge.start] = []
        outgoing_edges[edge.start] += [edge]

    return incoming_edges, outgoing_edges
