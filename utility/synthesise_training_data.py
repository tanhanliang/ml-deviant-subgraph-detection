"""
Contains function to edit training data.
"""
import random as r


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


def alter_process_cmdlines():
    """
    Queries the database for a certain pattern of graphs, then alters the cmdlines of
    all processes in that graph.

    :return: A list of tuples (label, graph). label is an integer, graph is a Graph object.
    """
    import make_training_data.format_training_data as format
    # training_data is a list of tuples (label, Graph)
    training_data = format.get_training_data()
    total_length = 11

    for i in range(len(training_data)):
        graph = training_data[i][1]
        for node_id in graph.nodes:
            node = graph.nodes[node_id]
            if "Process" in node.labels:
                if training_data[i][0] == 0:
                    target_word_idx = r.randint(0, 9)
                    new_cmd = get_rand_string(target_word_idx) + ' -k ' + get_rand_string(9-target_word_idx)
                    node.properties["cmdline"] = new_cmd
                else:
                    node.properties["cmdline"] = get_rand_string(10)

    return training_data
