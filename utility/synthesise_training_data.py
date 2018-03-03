"""
Contains function to edit training data.
"""
import random as r
import make_training_data.fetch_training_data as fetch
import make_training_data.format_training_data as format


def get_rand_string(length):
    """
    Generates a random string of length range comprising of random numbers, delimited by '/'.

    :param length: An integer
    :return: A string
    """
    accum = ""
    for i in range(0, length):
        accum += "/" + str(r.randint(101,999))

    return accum + "/"


def get_graphs_altered_cmdlines():
    """
    Queries the database for a certain pattern of graphs, then alters the cmdlines of
    all processes in that graph.

    :return: A list of tuples (label, graph). label is an integer, graph is a Graph object.
    """
    # training_data is a list of tuples (label, Graph)
    results = fetch.get_train_4_node_simple()
    training_graphs = format.label_and_process_data(results)
    total_length = 11

    for i in range(len(training_graphs)):
        graph = training_graphs[i][1]
        for node_id in graph.nodes:
            node = graph.nodes[node_id]
            if "Process" in node.labels:
                if training_graphs[i][0] == 0:
                    target_word_idx = r.randint(0, 9)
                    new_cmd = get_rand_string(target_word_idx) + ' -k ' + get_rand_string(9-target_word_idx)
                    node.properties["cmdline"] = new_cmd
                else:
                    node.properties["cmdline"] = get_rand_string(10)

    return training_graphs
