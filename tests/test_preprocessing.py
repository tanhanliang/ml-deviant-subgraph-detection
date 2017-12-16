"""
Tests for preprocessing functions
"""

import unittest
from data_processing.preprocessing import *


class MockNode:
    def __init__(self, node_id, properties=None):
        self.id = node_id

        if properties is None:
            self.properties = {}
        else:
            self.properties = properties


class MockEdge:
    def __init__(self, edge_id, start=-1, end=-1, edge_type='NoType'):
        self.id = edge_id
        self.start = start
        self.end = end
        self.type = edge_type


class MockPath:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.relationships = edges


class MockBoltStatementResult:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def data(self):
        return [{'path': MockPath(self.nodes, self.edges)}]


class TestPreprocessingFns(unittest.TestCase):
    def test_get_nodes_edges(self):
        nodes = [MockNode(1), MockNode(2), MockNode(3)]
        edges = [MockEdge(1), MockEdge(2), MockEdge(3)]
        data = MockBoltStatementResult(nodes, edges)
        nodes, edges = get_nodes_edges(data)

        for node_id in range(1, 4):
            self.assertEquals(id, nodes[node_id].id)
            self.assertEquals(id, edges[node_id].id)

    def test_build_adjacency_matrix(self):
        nodes = [MockNode(1), MockNode(2), MockNode(3), MockNode(4)]
        edges = [MockEdge(1, 1, 2), MockEdge(2, 2, 3), MockEdge(3, 3, 4)]
        data = MockBoltStatementResult(nodes, edges)
        adj_matrix = build_adjacency_matrix(data)
        shape = adj_matrix.matrix.shape

        self.assertEquals(shape, (len(nodes), len(nodes)))

        index_to_id = {}
        for node_id, index in adj_matrix.id_to_index.items():
            index_to_id[index] = node_id

        for row in range(shape[0]):
            for col in range(shape[1]):
                if adj_matrix.matrix[row, col] == 1:
                    start_id = index_to_id[row]
                    end_id = index_to_id[col]
                    self.assertTrue(end_id-start_id == 1)

    def test_consolidate_node_versions(self):
        nodes = [MockNode(1), MockNode(2), MockNode(3), MockNode(4)]
        edges = [MockEdge(1, 1, 2, 'PROC_OBJ_PREV'),
                 MockEdge(2, 2, 3, 'GLOB_OBJ_PREV'),
                 MockEdge(3, 3, 4, 'META_PREV')]
        incoming_edges, outgoing_edges = build_in_out_edges(edges)
        consolidate_node_versions(nodes, edges, incoming_edges, outgoing_edges)

        self.assertTrue(len(nodes) == 1)
        self.assertTrue(nodes[0].id == 1)
        self.assertTrue(len(edges) == 0)

def main():
    unittest.main()
