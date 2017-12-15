"""
Tests for preprocessing functions
"""

import unittest
from data_processing.preprocessing import *


class MockNode:
    def __init__(self, id):
        self.id = id


class MockEdge:
    def __init__(self, id):
        self.id = id


class MockPath:
    def __init__(self):
        self.nodes = [MockNode(1), MockNode(2), MockNode(3)]
        self.relationships = [MockEdge(1), MockEdge(2), MockEdge(3)]


class MockBoltStatementResult:
    def data(self):
        return [{'path': MockPath()}]


class TestPreprocessingFns(unittest.TestCase):
    def test_get_nodes_edges(self):
        data = MockBoltStatementResult()
        nodes, edges = get_nodes_edges(data)

        for id in range(1,4):
            self.assertEquals(id, nodes[id].id)
            self.assertEquals(id, edges[id].id)


def main():
    unittest.main()
