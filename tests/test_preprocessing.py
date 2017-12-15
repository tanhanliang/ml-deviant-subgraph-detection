"""
Tests for preprocessing functions
"""

import unittest
from data_processing.preprocessing import *
from data_processing.neo4j_interface_fns import *


class MockNode:
    def __init__(self, id):
        self.node_id = id

    def id(self):
        return self.node_id


class MockEdge:
    def __init__(self, id):
        self.edge_id = id

    def id(self):
        return self.edge_id


class MockPath:
    def nodes(self):
        return [MockNode(1), MockNode(2), MockNode(3)]

    def edges(self):
        return [MockEdge(1), MockEdge(2), MockEdge(3)]


class MockBoltStatementResult:
    def data(self):
        return {'path': MockPath()}


class TestPreprocessingFns(unittest.TestCase):
    def test_get_nodes_edges(self):
        data = MockBoltStatementResult()
        nodes, edges = get_nodes_edges(data)

        self.assertDictEqual(nodes, {1: MockNode(1), 2: MockNode(2), 3: MockNode(3)})
        self.assertDictEqual(edges, {1: MockEdge(1), 2: MockEdge(2), 3: MockEdge(3)})


def main():
    unittest.main()
