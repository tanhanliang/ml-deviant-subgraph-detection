"""This file contains functions to apply different hash functions to strings"""

import simhash


def hash_simhash(text):
    """
    Hashes the value using the SimHash algorithm.

    :param text: The string value to be hashed.
    :return: A hash value as an integer
    """

    return int(simhash.Simhash(text).value/100)


def hash_labels_prop(labels, node_label_hash, property):
    """
    Computes a hash value using SimHash to hash all the labels on the node, and
    concatenates that value with the SimHash of the string property provided.

    :param labels: A list of strings
    :param node_label_hash: A Dictionary of String -> integer
    Each label is assigned an integer which is a power of 2, to ensure that each combination
    of labels has a unique hash value.
    :param property: A string
    :return: An integer
    """

    hash_value = 0
    for label in labels:
        hash_value += node_label_hash[label]

    hash_value *= 1e10
    hash_value += int(str(hash_simhash(property))[:10])
    return hash_value
