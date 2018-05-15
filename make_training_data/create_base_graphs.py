from make_training_data.synthetic_graphs import FakeEdge, FakeNode
import data_processing.preprocessing as pre
import data_processing.graphs as graphs


def make_graph_16_nodes():
    """
    Creates a basic graph object with no properties set.

    :return: A Graph object
    """

    nodes = {}
    edges = {}

    for i in range(1, 17):
        nodes[i] = FakeNode(i)

    # Yes I have to manually assign all the edges...
    edges[1] = FakeEdge(1, 3, 1)
    edges[2] = FakeEdge(2, 4, 1)
    edges[3] = FakeEdge(3, 2, 1)
    edges[4] = FakeEdge(4, 4, 12)
    edges[5] = FakeEdge(5, 5, 3)
    edges[6] = FakeEdge(6, 7, 5)
    edges[7] = FakeEdge(7, 14, 7)
    edges[8] = FakeEdge(8, 6, 3)
    edges[9] = FakeEdge(9, 8, 6)
    edges[10] = FakeEdge(10, 13, 8)
    edges[11] = FakeEdge(11, 15, 8)
    edges[12] = FakeEdge(12, 6, 9)
    edges[13] = FakeEdge(13, 16, 9)
    edges[14] = FakeEdge(14, 9, 10)
    edges[15] = FakeEdge(15, 2, 11)

    incoming_edges, outgoing_edges = pre.build_in_out_edges(edges)
    graph = graphs.Graph(nodes, edges, incoming_edges, outgoing_edges)
    return graph


def make_graph_32_nodes():
    """
    Creates a basic graph object with no properties set.

    :return: A Graph object
    """

    nodes = {}
    edges = {}

    for i in range(1, 33):
        nodes[i] = FakeNode(i)

    # Yes I have to manually assign all the edges...
    edges[1] = FakeEdge(1, 3, 1)
    edges[2] = FakeEdge(2, 4, 1)
    edges[3] = FakeEdge(3, 2, 1)
    edges[4] = FakeEdge(4, 4, 12)
    edges[5] = FakeEdge(5, 5, 3)
    edges[6] = FakeEdge(6, 7, 5)
    edges[7] = FakeEdge(7, 14, 7)
    edges[8] = FakeEdge(8, 6, 3)
    edges[9] = FakeEdge(9, 8, 6)
    edges[10] = FakeEdge(10, 13, 8)
    edges[11] = FakeEdge(11, 15, 8)
    edges[12] = FakeEdge(12, 6, 9)
    edges[13] = FakeEdge(13, 16, 9)
    edges[14] = FakeEdge(14, 9, 10)
    edges[15] = FakeEdge(15, 2, 11)
    edges[16] = FakeEdge(16, 6, 17)
    edges[17] = FakeEdge(17, 6, 18)
    edges[18] = FakeEdge(18, 6, 19)
    edges[19] = FakeEdge(19, 6, 20)
    edges[20] = FakeEdge(20, 6, 21)
    edges[21] = FakeEdge(21, 6, 22)
    edges[22] = FakeEdge(22, 6, 23)
    edges[23] = FakeEdge(23, 6, 24)
    edges[24] = FakeEdge(24, 6, 25)
    edges[25] = FakeEdge(25, 6, 26)
    edges[26] = FakeEdge(26, 6, 27)
    edges[27] = FakeEdge(27, 6, 28)
    edges[28] = FakeEdge(28, 6, 29)
    edges[29] = FakeEdge(29, 6, 30)
    edges[30] = FakeEdge(30, 6, 31)
    edges[31] = FakeEdge(31, 6, 32)

    incoming_edges, outgoing_edges = pre.build_in_out_edges(edges)
    graph = graphs.Graph(nodes, edges, incoming_edges, outgoing_edges)
    return graph


def make_dataset(base_fn):
    training_graphs = []

    for i in range(1000):
        training_graphs.append((0, base_fn()))

    for i in range(1000):
        training_graphs.append((1, base_fn()))

    return training_graphs
