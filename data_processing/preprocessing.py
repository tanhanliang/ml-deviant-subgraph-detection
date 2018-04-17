"""
This module contains functions to pre-process(clean) graph data from Neo4j into a convenient form.
"""

import sys
import patchy_san.parameters as params
import data_processing.graphs as graphs

VERSION_TYPES = ['GLOB_OBJ_PREV', 'META_PREV', 'PROC_OBJ_PREV']


def get_graph(results):
    """
    Builds a graph object from a BoltStatementResult object which is the
    raw result of a neo4j query.

    :param results: A BoltStatementResult object describing all paths in the query
    :return: A Graph object
    """

    nodes = {}
    edges = {}

    for result in results.data():
        for path_name in result:
            for node in result[path_name].nodes:
                nodes[node.id] = node
            for edge in result[path_name].relationships:
                edges[edge.id] = edge

    incoming_edges, outgoing_edges = build_in_out_edges(edges)
    return graphs.Graph(nodes, edges, incoming_edges, outgoing_edges)


def get_graphs_by_result(results):
    """
    Builds a list of Graphs for every result in the provided
    BoltStatementResult. Also cleans the data. If the result does not lose any nodes as a result
    of the cleaning (the result is clean) then the nodes and edges which represent it are added
    to the list of tuples.

    :param results: A BoltStatementResult object describing all paths in the query
    :return: A list of Graphs
    """
    result_list = []
    deleted = 0

    for subgraph_dict in results.data():
        nodes = {}
        edges = {}
        for path in subgraph_dict:
            for node in subgraph_dict[path].nodes:
                nodes[node.id] = node
            for edge in subgraph_dict[path].relationships:
                edges[edge.id] = edge

        node_count = len(nodes)
        incoming_edges, outgoing_edges = build_in_out_edges(edges)
        graph = graphs.Graph(nodes, edges, incoming_edges, outgoing_edges)

        if params.CLEAN_TRAIN_DATA:
            clean_data(graph)

        if node_count == len(nodes):
            result_list.append(graph)
        else:
            deleted += 1
    print("Deleted: " + str(deleted))
    return result_list


def consolidate_node_versions(graph):
    """
    Given a Dictionary of node_id -> node and a Dictionary of edge_id -> edge,
    for all adjacent edges and nodes (node1)-[edge]->(node2) where
    type(edge) is a previous version type, removes edge and node2.
    All outgoing and incoming edges to node2 are glued to node1.

    Note: This may produce a graph where more than one edge of the same type may exist between
    2 nodes. Currently this is not a problem because the adjacency matrix produced from
    the set of edges and nodes cannot contain duplicated edges.

    :param graph: A Graph object
    :return: nothing
    """
    nodes = graph.nodes
    edges = graph.edges
    incoming_edges = graph.incoming_edges
    outgoing_edges = graph.outgoing_edges

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
                    # This assumes that there are no nodes connected by two edges in
                    # both directions
                    outgoing_edges[master_node_id] += [outgoing_edge]
                    outgoing_edges.pop(removed_node_id)

            if removed_node_id in incoming_edges.keys():
                for incoming_edge in incoming_edges[removed_node_id]:
                    if incoming_edge.start is not master_node_id:
                        incoming_edge.end = master_node_id
                        # The key master_node_id may not exist
                        if master_node_id not in incoming_edges:
                            incoming_edges[master_node_id] = []
                            incoming_edges[master_node_id] += [incoming_edge]
                            incoming_edges.pop(removed_node_id)

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
        incoming_edges[edge.end].append(edge)

        if not outgoing_edges.__contains__(edge.start):
            outgoing_edges[edge.start] = []
        outgoing_edges[edge.start].append(edge)

    for edge_dict in [incoming_edges, outgoing_edges]:
        for edge_id in edge_dict:
            edges = edge_dict[edge_id]
            edge_dict[edge_id] = sorted(edges, key=lambda x: x.id)

    return incoming_edges, outgoing_edges


def remove_anomalous_nodes_edges(graph):
    """
    Removes all nodes from the Dictionaries of node_id -> node and edge_id -> edge
    where node.anomalous = true. This is an artefact of the provenance capture process
    and should not be included.

    Also removes nodes which do not have a timestamp, as this is also anomalous.

    :param graph: A Graph object
    :return: nothing
    """

    for node_id in list(graph.nodes.keys()):
        node_prop = graph.nodes[node_id].properties
        if node_prop.__contains__('anomalous') and node_prop['anomalous'] or \
                'timestamp' not in node_prop:
            pop_related_edges(graph, node_id, True)
            pop_related_edges(graph, node_id, False)
            graph.nodes.pop(node_id)


def pop_related_edges(graph, node_id, is_incoming):
    """
    Pops every edge_id -> edge mapping in a Dictionary where either the start or the
    end of the edge has node_id. Also removes mapping from node_id -> edge from the given
    node_edge_dict if it exists.

    :param graph: A Graph object
    :param node_id: An Integer id number of a node of which all related edges will be deleted
    :return: nothing
    """

    if is_incoming:
        node_edge_dict = graph.incoming_edges
    else:
        node_edge_dict = graph.outgoing_edges

    if node_id in node_edge_dict:
        for edge in node_edge_dict[node_id]:
            if is_incoming:
                other_id = edge.start
                new_outgoing_edges = [edg for edg in graph.outgoing_edges[other_id] if edge.id != edg.id]
                graph.outgoing_edges[other_id] = new_outgoing_edges

            else:
                other_id = edge.end
                new_incoming_edges = [edg for edg in graph.incoming_edges[other_id] if edge.id != edg.id]
                graph.incoming_edges[other_id] = new_incoming_edges

            if edge.id in node_edge_dict:
                graph.edges.pop(edge.id)
        node_edge_dict.pop(node_id)


def group_nodes_by_uuid(graph):
    """
    Creates a dictionary of uuid -> Set of nodes. The set contains all nodes with that uuid.
    Nodes with no uuid are not included.

    :param graph: A Graph object
    :return: A Dictionary of uuid -> Set of nodes
    """

    uuid_to_nodes = {}

    for _, node in graph.nodes.items():
        node_prop = node.properties
        if 'uuid' in node_prop:
            uuid = node_prop['uuid']
            if uuid not in uuid_to_nodes:
                uuid_to_nodes[uuid] = set()
            uuid_to_nodes[uuid].add(node)

    return uuid_to_nodes


def rename_symlinked_files_timestamp(graph):
    """
    For all file nodes with the same uuid, renames them with the name of the node
    with the smallest timestamp (oldest node). Nodes with no name are not renamed.
    Nodes with no timestamp are still renamed (all nodes should have a timestamp).
    If multiple nodes have the same timestamp, a name is selected arbitrarily.

    :param graph: A Graph object
    :return: Nothing
    """

    uuid_to_nodes = group_nodes_by_uuid(graph)

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


def remove_duplicate_edges(graph):
    """
    Checks every node's incoming and outgoing edges, and if two edges have the same
    start (for incoming edges) or end (for outgoing edges) nodes, they are removed.

    This method assumes that two edges between the same 2 nodes will always be of the same type,
    and therefore can be removed. I believe that this is a valid assumption due to the
    nature of the provenance data. Edges between the same 2 nodes only exist because of
    the node version consolidation step above, which deletes nodes representing past versions.
    If nodes representing past versions have a common neighbor, then the edge type between them
    and the neighbor are all the same.

    :param graph: A Graph object
    :return: nothing
    """

    edges = graph.edges
    outgoing_edges = graph.outgoing_edges

    for node_id in outgoing_edges.keys():
        edge_end_ids = set()
        for edge in outgoing_edges[node_id]:
            if edge.end not in edge_end_ids:
                edge_end_ids.add(edge.end)
            else:
                outgoing_edges[node_id].remove(edge)
                edges.pop(edge.id)


def clean_data_raw(results):
    """
    Given a BoltStatementResult object, cleans the data by removing anomalous nodes,
    consolidating node versions and renaming symlinked files.

    :param results: A BoltStatementResult
    :return: A tuple of (nodes, edges). nodes is a Dictionary of node_id -> node, edges
    is a Dictionary of edge_id -> edge
    """

    graph = get_graph(results)
    return clean_data(graph)


def clean_data(graph):
    """
    Cleans the data by removing anomalous nodes,
    consolidating node versions and renaming symlinked files.

    :param graph: A Graph object
    :return: Nothing
    """

    consolidate_node_versions(graph)
    remove_duplicate_edges(graph)
    remove_anomalous_nodes_edges(graph)
    rename_symlinked_files_timestamp(graph)
