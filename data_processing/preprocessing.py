"""
This module contains functions to pre-process(clean) graph data from Neo4j into a convenient form.
"""

import numpy as np
import sys
from data_processing.adj_matrices import AdjacencyMatrix

VERSION_TYPES = ['GLOB_OBJ_PREV', 'META_PREV', 'PROC_OBJ_PREV']


def get_nodes_edges(results):
    """
    Builds a Dictionary of node_id -> node and a Dictionary of edge_id -> edge from a
    BoltStatementResult object which is the raw result of a neo4j query.

    :param results: A BoltStatementResult object describing all paths in the query
    :return: A tuple of (Dictionary, Dictionary)
    """

    nodes = {}
    edges = {}

    for result in results.data():
        for node in result['path'].nodes:
            nodes[node.id] = node
        for edge in result['path'].relationships:
            edges[edge.id] = edge

    return nodes, edges


def build_adjacency_matrix(results):
    """
    Builds an adjacency matrix based on a given graph, represented as a BoltStatementResult

    :param results: A BoltStatementResult describing all paths within the graph
    :return: An adjacency matrix representation for the graph, using a np.matrix
    """

    nodes, edges = get_nodes_edges(results)

    incoming_edges, outgoing_edges = build_in_out_edges(edges)
    consolidate_node_versions(nodes, edges, incoming_edges, outgoing_edges)
    remove_anomalous_nodes_edges(nodes, edges, incoming_edges, outgoing_edges)

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


def consolidate_node_versions(nodes, edges, incoming_edges, outgoing_edges):
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
    :param incoming_edges: A Dictionary of node_id -> list of edges (incoming edges to that node)
    :param outgoing_edges: A Dictionary of node_id -> list of edges (outgoing edges from that node)
    :return: nothing
    """

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


def build_in_out_edges(edges):
    """
    Given a Dictionary of edge_id -> edge, builds two
    dictionaries of node_id -> edge (incoming or outgoing edges from that node).

    :param edges: A Dictionary of edge_id -> edge
    :return: (incoming_edges, outgoing_edges), a tuple of Dictionaries of node_id to
    list of incoming/outgoing edges to/from that node
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


def remove_anomalous_nodes_edges(nodes, edges, incoming_edges, outgoing_edges):
    """
    Removes all nodes from the Dictionaries of node_id -> node and edge_id -> edge
    where node.anomalous = true. This is an artefact of the provenance capture process
    and should not be included.

    :param nodes: A Dictionary of node_id -> node
    :param edges: A Dictionary of edge_id -> edge
    :param incoming_edges: A Dictionary of node_id -> list of edges (incoming edges to that node)
    :param outgoing_edges: A Dictionary of node_id -> list of edges (outgoing edges from that node)
    :return: nothing
    """

    for node_id in list(nodes.keys()):
        node_prop = nodes[node_id].properties
        if node_prop.__contains__('anomalous') and node_prop['anomalous']:
            pop_related_edges(incoming_edges, edges, node_id)
            pop_related_edges(outgoing_edges, edges, node_id)
            nodes.pop(node_id)


def pop_related_edges(node_edge_dict, edges, node_id):
    """
    Pops every edge_id -> edge mapping in a Dictionary where either the start or the
    end of the edge has node_id. Also removes mapping from node_id -> edge from the given
    node_edge_dict if it exists.

    :param node_edge_dict: A Dictionary of node_id -> edge (incoming or outgoing edges from that node)
    :param edges: A Dictionary of edge_id -> edge
    :param node_id: An Integer id number of a node of which all related edges will be deleted
    :return: nothing
    """

    if node_edge_dict.__contains__(node_id):
        for edge in node_edge_dict[node_id]:
            if edges.__contains__(edge.id):
                edges.pop(edge.id)
        node_edge_dict.pop(node_id)


def group_nodes_by_uuid(nodes):
    """
    Creates a dictionary of uuid -> Set of nodes. The set contains all nodes with that uuid.
    Nodes with no uuid are not included.

    :param nodes: A Dictionary of node_id -> node
    :return: A Dictionary of uuid -> Set of nodes
    """

    uuid_to_nodes = {}

    for _, node in nodes.items():
        node_prop = node.properties
        if 'uuid' in node_prop:
            uuid = node_prop['uuid']
            if uuid not in uuid_to_nodes:
                uuid_to_nodes[uuid] = set()
            uuid_to_nodes[uuid].add(node)

    return uuid_to_nodes


def rename_symlinked_files_timestamp(nodes):
    """
    For all file nodes with the same uuid, renames them with the name of the node
    with the smallest timestamp (oldest node). Nodes with no name are not renamed.
    Nodes with no timestamp are still renamed (all nodes should have a timestamp).
    If multiple nodes have the same timestamp, a name is selected arbitrarily.

    :param nodes: A Dictionary of node_id -> node
    :return: Nothing
    """

    uuid_to_nodes = group_nodes_by_uuid(nodes)

    for uuid, nodes_set in uuid_to_nodes.items():
        if len(nodes_set) > 1:
            timestamp = sys.maxsize
            smallest_ts_node = None

            for node in nodes_set:
                node_prop = node.properties
                if 'timestamp' in node_prop and 'name' in node_prop \
                        and node_prop['timestamp'] < timestamp:
                    timestamp = node_prop['timestamp']
                    smallest_ts_node = node

            if smallest_ts_node is not None:
                for node in nodes_set:
                    if 'name' in node.properties:
                        # TODO: Decide what to do for multiple names
                        node.properties['name'] = smallest_ts_node.properties['name']


def clean_data(results):
    """
    Given a BoltStatementResult object, cleans the data by removing anomalous nodes,
    consolidating node versions and renaming symlinked files.

    :param results: A BoltstatementResult object
    :return: A tuple of (nodes, edges). nodes is a Dictionary of node_id -> node, edges
    is a Dictionary of edge_id -> edge
    """

    nodes, edges = get_nodes_edges(results)
    incoming_edges, outgoing_edges = build_in_out_edges(edges)

    consolidate_node_versions(nodes, edges, incoming_edges, outgoing_edges)
    remove_anomalous_nodes_edges(nodes, edges, incoming_edges, outgoing_edges)
    rename_symlinked_files_timestamp(nodes)

    return nodes, edges
