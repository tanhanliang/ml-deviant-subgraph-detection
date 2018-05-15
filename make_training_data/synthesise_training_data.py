"""
Contains function to edit training data.
"""
import random as r
import make_training_data.fetch_training_data as fetch
import make_training_data.format_training_data as fmt
import patchy_san.parameters as params
import patchy_san.make_cnn_input as make


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


def get_graphs_altered_cmdlines(cmdline_len, simple=False):
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

        for edge in graph.edges.values():
            edge.type = "COMM"
            edge.properties["state"] = "READ"

        for node_id in graph.nodes:
            node = graph.nodes[node_id]
            node.properties["name"] = ["/standard/name"]

            if node == chosen_node:
                # pattern of interest
                if training_graphs[i][0] == 0:
                    if simple:
                        node.properties["cmdline"] = ' -k ' + get_rand_string(cmdline_len - 1)
                    else:
                        target_word_idx = r.randint(0, cmdline_len-1)
                        new_cmd = get_rand_string(target_word_idx) + ' -k ' + get_rand_string(cmdline_len-target_word_idx-1)
                        node.properties["cmdline"] = new_cmd

                # negative data/adversarial pattern
                else:
                    node.properties["cmdline"] = get_rand_string(cmdline_len)
            else:
                node.properties["cmdline"] = get_rand_string(cmdline_len)

    return training_graphs


def get_graphs_test_negative_data_6():
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
            if incoming_edge_count == 3:
                node.labels = {"Process"}
            elif incoming_edge_count == 2:
                node.labels = {"Pipe"}
            else:
                node.labels = {"File", "Global"}

            node.properties["cmdline"] = pattern_cmdline
            node.properties["name"] = [pattern_name]

        # Set the edges
        for edge_id in graph.edges:
            edge = graph.edges[edge_id]
            start_node = graph.nodes[edge.start]
            end_node = graph.nodes[edge.end]

            if "Pipe" in start_node.labels and "Process" in end_node.labels:
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


def get_graphs_test_negative_data_4():
    """
    Builds training data for a 2-class classification problem. The negative data is split into 3 parts,
    with tweaks to either a node label, a node property or a edge state compared to the pattern of interest.

    :return: A list of tuples (label, graph). label is an integer, graph is a Graph object.
    """

    results = fetch.get_train_4_node_simple()
    training_graphs = fmt.label_and_process_data(results)

    pattern_cmdline = "/My/name/is/Homer/Simpson/"
    pattern_name = "/super/secret/password/database/pwd.db"

    # manually set relevant properties
    for (label, graph) in training_graphs:
        # Set data for both classes to be the same first. Don't check label yet.
        # Set the node labels and node properties
        socket_set = False

        for node_id in graph.nodes:
            if node_id in graph.incoming_edges:
                incoming_edge_count = len(graph.incoming_edges[node_id])
            else:
                incoming_edge_count = 0

            node = graph.nodes[node_id]
            if incoming_edge_count == 3:
                node.labels = {"Process"}
            elif not socket_set:
                node.labels = {"Socket"}
                socket_set = True
            else:
                node.labels = {"File", "Global"}

            node.properties["cmdline"] = pattern_cmdline
            node.properties["name"] = [pattern_name]

        # Set the edges
        for edge_id in graph.edges:
            edge = graph.edges[edge_id]
            start_node = graph.nodes[edge.start]
            end_node = graph.nodes[edge.end]

            if "Socket" in start_node.labels and "Process" in end_node.labels:
                edge.type = "COMM"
                edge.properties["state"] = "SERVER"
            else:
                edge.type = "PROC_OBJ"
                edge.properties["state"] = "RaW"

        # Now we tweak things if label is 1 (the negative data)
        if label == 1:
            choice = r.randint(2,3)

            if choice == 1:
                tweak_node = None
                for node in graph.nodes.values():
                    if "Socket" in node.labels:
                        tweak_node = node
                        break

                assert(tweak_node is not None)

                tweak_node.labels = {"File", "Global"}

            elif choice == 2:
                tweak_node = None
                for node in graph.nodes.values():
                    if "File" in node.labels:
                        tweak_node = node
                        break

                assert(tweak_node is not None)

                tweak_node.properties["name"] = ["test/"*10]

            else:
                tweak_edge = None
                for edge in graph.edges.values():
                    start_node = graph.nodes[edge.start]
                    end_node = graph.nodes[edge.end]
                    if "File" in start_node.labels and "Process" in end_node.labels:
                        tweak_edge = edge
                        break

                assert(tweak_edge is not None)

                tweak_edge.properties["state"] = "WRITE"

    return training_graphs


def get_graphs_test_negative_data_4_easy():
    """
    Builds training data for a 2-class classification problem. The negative data is completely different from
    the pattern of interest.

    :return: A list of tuples (label, graph). label is an integer, graph is a Graph object.
    """

    results = fetch.get_train_4_node_simple()
    training_graphs = fmt.label_and_process_data(results)

    pattern_cmdline = "/My/name/is/Homer/Simpson/"
    pattern_name = "/super/secret/password/database/pwd.db"
    possible_labels = list(params.NODE_TYPE_HASH.keys())
    possible_edge_types = list(make.EDGE_TYPE_HASH.keys())
    possible_edges_states = list(make.EDGE_STATE_HASH.keys())

    # manually set relevant properties
    for (label, graph) in training_graphs:
        # Set data for both classes to be the same first. Don't check label yet.
        # Set the node labels and node properties
        if label == 0:
            socket_set = False

            for node_id in graph.nodes:
                if node_id in graph.incoming_edges:
                    incoming_edge_count = len(graph.incoming_edges[node_id])
                else:
                    incoming_edge_count = 0

                node = graph.nodes[node_id]
                if incoming_edge_count == 3:
                    node.labels = {"Process"}
                elif not socket_set:
                    node.labels = {"Socket"}
                    socket_set = True
                else:
                    node.labels = {"File", "Global"}

                node.properties["cmdline"] = pattern_cmdline
                node.properties["name"] = [pattern_name]

            # Set the edges
            for edge_id in graph.edges:
                edge = graph.edges[edge_id]
                start_node = graph.nodes[edge.start]
                end_node = graph.nodes[edge.end]

                if "Socket" in start_node.labels and "Process" in end_node.labels:
                    edge.type = "COMM"
                    edge.properties["state"] = "SERVER"
                else:
                    edge.type = "PROC_OBJ"
                    edge.properties["state"] = "RaW"

        # Now we tweak things if label is 1 (the negative data)
        if label == 1:
            for node_id in graph.nodes:
                node = graph.nodes[node_id]
                choice = r.randint(0, len(possible_labels)-1)
                node.labels = {possible_labels[choice]}
                node.properties["cmdline"] = get_rand_string(10)
                node.properties["name"] = [get_rand_string(10)]

            # Set the edges
            for edge in graph.edges.values():
                type_choice = r.randint(0, len(possible_edge_types)-1)
                state_choice = r.randint(0, len(possible_edges_states)-1)

                edge.type = possible_edge_types[type_choice]
                edge.properties["state"] = possible_edges_states[state_choice]

    return training_graphs


def get_graphs_n_nodes_hard(training_graphs):
    """
    Assigns some arbitrary properties to nodes and edges in the graphs provided.

    :param results:
    :return:
    """
    possible_labels = list(params.NODE_TYPE_HASH.keys())
    possible_edge_types = list(make.EDGE_TYPE_HASH.keys())
    possible_edges_states = list(make.EDGE_STATE_HASH.keys())

    # manually set relevant properties
    for (label, graph) in training_graphs:
        choice = r.randint(0, params.MAX_NODES-1)
        chosen_node = list(graph.nodes.values())[choice]

        for node in graph.nodes.values():
            label_choice = r.randint(0, len(possible_labels)-1)

            node.labels = {possible_labels[label_choice]}
            node.properties["cmdline"] = get_rand_string(10)

            if node == chosen_node and label == 0:
                position = r.randint(0, params.EMBEDDING_LENGTH-1)
                new_name = get_rand_string(position) + " hi " + get_rand_string(params.EMBEDDING_LENGTH-position-1)
                node.properties["name"] = [new_name]
            else:
                node.properties["name"] = [get_rand_string(10)]

        # Set the edges
        for edge in graph.edges.values():
            type_choice = r.randint(0, len(possible_edge_types)-1)
            state_choice = r.randint(0, len(possible_edges_states)-1)

            edge.type = possible_edge_types[type_choice]
            edge.properties["state"] = possible_edges_states[state_choice]

    return training_graphs


def get_graphs_n_nodes_easy(training_graphs):
    """
    Assigns some arbitrary properties to nodes and edges in the graphs provided.

    :param results:
    :return:
    """
    possible_labels = list(params.NODE_TYPE_HASH.keys())
    possible_edge_types = list(make.EDGE_TYPE_HASH.keys())
    possible_edges_states = list(make.EDGE_STATE_HASH.keys())

    # manually set relevant properties
    for (label, graph) in training_graphs:
        # Set data for both classes to be the same first. Don't check label yet.
        # Set the node labels and node properties
        if label == 0:
            for node_id in graph.nodes:
                if node_id in graph.incoming_edges:
                    incoming_edge_count = len(graph.incoming_edges[node_id])
                else:
                    incoming_edge_count = 0

                node = graph.nodes[node_id]

                if incoming_edge_count == 0:
                    node.labels = {"File", "Global"}
                    node.properties["cmdline"] = "/11/22/33/44/55/66/77/88/99/10"
                    node.properties["name"] = ["/a/b/c/d/e/f/g/h/i/j10"]

                elif incoming_edge_count == 1:
                    node.labels = {"Meta"}
                    node.properties["cmdline"] = "/11/22/33/44/55/66/77/88/99/10"
                    node.properties["name"] = ["/a/b/c/d/e/f/g/h/i/j10"]

                elif incoming_edge_count == 2:
                    node.labels = {"Socket"}
                    node.properties["cmdline"] = "/111/222/333/444/555/666/777/888/999/100"
                    node.properties["name"] = ["/aa/bb/cc/dd/ee/ff/gg/hh/ii/jj10"]

                elif incoming_edge_count == 3:
                    node.labels = {"Process"}
                    node.properties["cmdline"] = "/one/two/three/four/five/six/seven/eight/nine/ten"
                    node.properties["name"] = ["/1/2/3/4/5/6/7/8/9/10"]

            # Set the edges
            for edge_id in graph.edges:
                edge = graph.edges[edge_id]

                if edge.start in graph.incoming_edges:
                    start_node_incoming_count = len(graph.incoming_edges[edge.start])

                else:
                    start_node_incoming_count = 0

                if start_node_incoming_count == 0:
                    edge.type = "PROC_OBJ"
                    edge.properties["state"] = "RaW"

                elif start_node_incoming_count == 1:
                    edge.type = "PROC_PARENT"
                    edge.properties["state"] = "BIN"

                elif start_node_incoming_count == 2:
                    edge.type = "META_PREV"
                    edge.properties["state"] = "READ"

                else:
                    edge.type = "GLOB_OBJ_PREV"
                    edge.properties["state"] = "CLIENT"

        if label == 1:
            for node in graph.nodes.values():
                label_choice = r.randint(0, len(possible_labels)-1)

                node.labels = {possible_labels[label_choice]}
                node.properties["cmdline"] = get_rand_string(10)
                node.properties["name"] = [get_rand_string(10)]

            # Set the edges
            for edge in graph.edges.values():
                type_choice = r.randint(0, len(possible_edge_types)-1)
                state_choice = r.randint(0, len(possible_edges_states)-1)

                edge.type = possible_edge_types[type_choice]
                edge.properties["state"] = possible_edges_states[state_choice]

    return training_graphs


def get_graphs_test_negative_data_n(training_graphs):
    """
    Builds training data for a 2-class classification problem. The negative data is split into 3 parts,
    with tweaks to either a node label, a node property or a edge state compared to the pattern of interest.

    :return: A list of tuples (label, graph). label is an integer, graph is a Graph object.
    """

    training_graphs = randomise_graph(training_graphs)

    # manually set relevant properties
    for (label, graph) in training_graphs:
        # Now we tweak things if label is 1 (the negative data)
        if label == 1:
            choice = r.randint(1,3)

            if choice == 1 or choice == 2:
                tweak_node = None
                for node in graph.nodes.values():
                    incoming = len(graph.incoming_edges[node.id])

                    if incoming == 3:
                        tweak_node = node
                        break

                assert(tweak_node is not None)

                if choice == 1:
                    tweak_node.labels = {"File", "Global"}
                else:
                    tweak_node.properties["name"] = ["/1/2/3/4/5/6/7/8/9/10"]

            elif choice == 3:
                tweak_edge = None
                for edge in graph.edges.values():
                    start_node = graph.nodes[edge.start]
                    start_incoming = len(graph.incoming_edges[start_node.id])
                    if start_incoming == 2:
                        tweak_edge = edge

                assert(tweak_edge is not None)

                tweak_edge.properties["state"] = "BIN"
                tweak_edge.type = "GLOB_OBJ_PREV"

    return training_graphs


def randomise_graph(training_graphs):
    possible_labels = list(params.NODE_TYPE_HASH.keys())
    possible_edge_types = list(make.EDGE_TYPE_HASH.keys())
    possible_edges_states = list(make.EDGE_STATE_HASH.keys())

    for (label, graph) in training_graphs:
        for node in graph.nodes.values():
            label_choice = r.randint(0, len(possible_labels)-1)

            node.labels = {possible_labels[label_choice]}
            node.properties["cmdline"] = get_rand_string(10)
            node.properties["name"] = [get_rand_string(10)]

        # Set the edges
        for edge in graph.edges.values():
            type_choice = r.randint(0, len(possible_edge_types)-1)
            state_choice = r.randint(0, len(possible_edges_states)-1)

            edge.type = possible_edge_types[type_choice]
            edge.properties["state"] = possible_edges_states[state_choice]

    return training_graphs
