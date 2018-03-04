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
