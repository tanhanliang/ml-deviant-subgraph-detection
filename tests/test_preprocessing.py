"""
Tests for preprocessing functions
"""

import unittest
import data_processing.preprocessing as pre
from data_processing.graphs import Graph


class MockNode:
    def __init__(self, node_id, properties=None):
        self.id = node_id

        if properties is None:
            self.properties = {}
        else:
            self.properties = properties
        self.properties["timestamp"] = 1000+node_id


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
        node_list = [MockNode(1), MockNode(2), MockNode(3)]
        edge_list = [MockEdge(1), MockEdge(2), MockEdge(3)]
        data = MockBoltStatementResult(node_list, edge_list)
        graph = pre.get_graph(data)

        for node_id in range(1, 4):
            self.assertEquals(node_id, graph.nodes[node_id].id)
            self.assertEquals(node_id, graph.edges[node_id].id)

    def test_consolidate_node_versions(self):
        nodes = {1: MockNode(1), 2: MockNode(2), 3: MockNode(3), 4: MockNode(4)}
        edges = {1: MockEdge(1, 1, 2, 'PROC_OBJ_PREV'),
                 2: MockEdge(2, 2, 3, 'GLOB_OBJ_PREV'),
                 3: MockEdge(3, 3, 4, 'META_PREV')}
        incoming_edges, outgoing_edges = pre.build_in_out_edges(edges)
        graph = Graph(nodes, edges, incoming_edges, outgoing_edges)
        pre.consolidate_node_versions(graph)

        self.assertTrue(len(graph.nodes) == 1)
        self.assertTrue(graph.nodes[1].id == 1)
        self.assertTrue(len(graph.edges) == 0)

    def test_build_in_out_edges(self):
        edges = {1: MockEdge(1, 1, 2), 2: MockEdge(2, 1, 3), 3: MockEdge(3, 1, 4)}
        incoming_edges, outgoing_edges = pre.build_in_out_edges(edges)

        self.assertTrue(len(outgoing_edges.keys()) == 1)
        self.assertTrue(len(incoming_edges.keys()) == 3)
        self.assertTrue(len(outgoing_edges[1]) == 3)
        self.assertTrue(incoming_edges[2][0].id == 1)
        self.assertTrue(incoming_edges[3][0].id == 2)
        self.assertTrue(incoming_edges[4][0].id == 3)

    def test_remove_anomalous_nodes(self):
        nodes = {1: MockNode(1, {'anomalous': True}), 2: MockNode(2, {'anomalous': True}),
                 3: MockNode(3, {'anomalous': False}), 4: MockNode(4, {'anomalous': False})}
        edges = {1: MockEdge(1, 1, 2), 2: MockEdge(2, 2, 3), 3: MockEdge(3, 3, 4)}
        incoming_edges, outgoing_edges = pre.build_in_out_edges(edges)
        graph = Graph(nodes, edges, incoming_edges, outgoing_edges)
        pre.remove_anomalous_nodes_edges(graph)

        self.assertTrue(len(graph.nodes) == 2)
        self.assertTrue(len(graph.edges) == 1)
        self.assertTrue(graph.edges[3].start == 3 and edges[3].end == 4)
        self.assertTrue(graph.nodes[3].id == 3 and nodes[4].id == 4)

    def test_rename_symlinked_files_timestamp(self):
        nodes = {1: MockNode(1, {'uuid': 10, 'timestamp': 100, 'name': '/etc/lib.so.6'}),
                 2: MockNode(2, {'uuid': 10, 'timestamp': 101, 'name': '/etc/lib.so.1234'}),
                 3: MockNode(3, {'uuid': 11, 'timestamp': 99, 'name': '/var/test'})}
        graph = Graph(nodes, {}, {}, {})
        pre.rename_symlinked_files_timestamp(graph)

        self.assertTrue(len(graph.nodes) == 3)
        self.assertTrue(graph.nodes[1].properties['name'] == '/etc/lib.so.6')
        self.assertTrue(graph.nodes[2].properties['name'] == '/etc/lib.so.6')
        self.assertTrue(graph.nodes[3].properties['name'] == '/var/test')


def main():
    unittest.main()
