"""
This module contains functions to extract and process data from a neo4j database.

Dependencies: neo4j-driver
"""
from neo4j.v1 import GraphDatabase, basic_auth
import numpy as np

def getSubGraphPaths(root_id, end_id):
    """
    Queries the neo4j database for all paths starting from a root node and ending
    at an end node

    :param root_id: The database Id of the root node
    :param end_id: The database Id of the end node
    :return: A BoltStatementResult object describing all paths between root node
    specified by uuid1, and end node specified by uuid2.
    """
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "neo4j"))
    session = driver.session()

    query = """
    MATCH path=(n)-[r*]->(m)
    WHERE Id(n) = $id1 AND Id(m) = $id2
    RETURN path
    """
    results = session.run(query, {"id1" : root_id, "id2" : end_id})
    session.close()
    return results

def buildAdjacencyMatrix(results):
    """
    Builds an adjacency matrix based on a given graph, represented as a BoltStatementResult

    :param results: A BoltStatementResult describing all paths within the graph
    :return: An adjacency matrix representation fo the graph, using a np.matrix
    """
    nodes = set()
    edges = set()

    for result in results.data():
        for node in result['path'].nodes:
            nodes.add(node)
        for edge in result['path'].relationships:
            edges.add(edge)

    node_count = len(nodes)
    adjacency_matrix = np.matrix(np.zeros(shape=(node_count, node_count), dtype=np.int8))
    id_to_index = {}

    idx = 0
    for node in nodes:
        id_to_index[node.id] = idx
        idx += 1

    for edge in edges:
        startIdx = id_to_index[edge.start]
        endIdx = id_to_index[edge.end]
        adjacency_matrix[startIdx, endIdx] = 1

    return AdjacencyMatrix(adjacency_matrix, id_to_index)

class AdjacencyMatrix:
    """Wrapper class to represent an Adjacency Matrix"""
    def __init__(self, matrix, id_to_index):
        """
        Initialises the matrix and dictionary containing ids

        self.matrix: A 2-dimensional numpy matrix
        self.labels: A dictionary of node id -> matrix index
        """
        self.matrix = matrix
        self.id_to_index = id_to_index

