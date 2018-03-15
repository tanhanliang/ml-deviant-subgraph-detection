"""
Contains function to edit training data.
"""
import random as r
import make_training_data.fetch_training_data as fetch
import make_training_data.format_training_data as fmt


def get_rand_string(length):
    """
    Generates a random string of length range comprising of random numbers, delimited by '/'.

    :param length: An integer
    :return: A string
    """
    accum = ""
    for i in range(0, length):
        accum += "/" + str(r.randint(101, 999))

    return accum + "/"


def get_graphs_altered_cmdlines(cmdline_len):
    """
    Queries the database for a certain pattern of graphs, then alters the cmdlines of
    all processes in that graph.

    :param cmdline_len: The length of the generated cmdline, in words delimited by punctuation.
    :return: A list of tuples (label, graph). label is an integer, graph is a Graph object.
    """
    # training_data is a list of tuples (label, Graph)
    results = fetch.get_train_4_node_simple()
    training_graphs = fmt.label_and_process_data(results)

    for i in range(len(training_graphs)):
        graph = training_graphs[i][1]
        rand_node = r.randint(0, len(graph.nodes)-1)
        chosen_node = list(graph.nodes.values())[rand_node]

        for node_id in graph.nodes:
            node = graph.nodes[node_id]
            node.properties["name"] = ["/standard/name"]

            if node == chosen_node:
                if training_graphs[i][0] == 0:
                    target_word_idx = r.randint(0, cmdline_len-1)
                    new_cmd = get_rand_string(target_word_idx) + ' -k ' + get_rand_string(cmdline_len-target_word_idx-1)
                    node.properties["cmdline"] = new_cmd
                else:
                    node.properties["cmdline"] = get_rand_string(cmdline_len)
            else:
                node.properties["cmdline"] = get_rand_string(cmdline_len)

    return training_graphs


def get_graphs_test_negative_data():
    """
    This builds data which tests if the model can use all 3 inputs on their own to make a
    classification.

    First, a general graph pattern is queried for. Each graph is then edited to a set pattern,
    to create the first class. The negative data is built by using a copy of this data, then making
    a slight change either to an edge, a node label, a cmdline on a node, or a name of a node.

    :return: A list of tuples (label, graph). label is an integer, graph is a Graph object.
    """

    results = fetch.get_train_6_node_general()
    training_graphs = fmt.label_and_process_data(results)

    pattern_cmdline = "/My/name/is/Homer/Simpson/"
    pattern_name = "/super/secret/password/database/pwd.db"

    for (label, graph) in training_graphs:
        # Set data for both classes to be the same first. Don't check label yet.
        # Set the node labels and node properties
        for node_id in graph.nodes:
            if node_id in graph.incoming_edges:
                incoming_edge_count = len(graph.incoming_edges[node_id])
            else:
                incoming_edge_count = 0

            node = graph.nodes[node_id]
            if incoming_edge_count == 3 or incoming_edge_count == 2:
                node.labels = {"Process"}
            else:
                node.labels = {"File", "Global"}

            node.properties["cmdline"] = pattern_cmdline
            node.properties["name"] = pattern_name

        # Set the edges
        for edge_id in graph.edges:
            edge = graph.edges[edge_id]
            start_node = graph.nodes[edge.start]
            end_node = graph.nodes[edge.end]

            if "Process" in start_node.labels and "Process" in end_node.labels:
                edge.type = "PROC_PARENT"
                edge.properties["state"] = "NONE"
            else:
                edge.type = "PROC_OBJ"
                edge.properties["state"] = "RaW"

        # Now we tweak things if label is 1 (the negative data)
        if label == 1:
            choice = r.randint(1,3)

            # Get a random node
            nodes_list = list(graph.nodes.values())
            tweak_node_idx = r.randint(0, len(nodes_list) - 1)
            tweaked_node = nodes_list[tweak_node_idx]

            # Get a random edge
            edges_list = list(graph.edges.values())
            tweak_edge_idx = r.randint(0, len(edges_list) - 1)
            tweaked_edge = edges_list[tweak_edge_idx]

            if choice == 1:
                tweaked_node.labels = {'Socket'}

            elif choice == 2:
                tweaked_node.properties["cmdline"] = "/My/name/is/Homer/Thompson/"

            else:
                tweaked_edge.properties["state"] = "SERVER"

    return training_graphs
