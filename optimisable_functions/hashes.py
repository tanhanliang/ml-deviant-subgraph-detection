"""This file contains functions to apply different hash functions to strings"""

import simhash


def hash_simhash(**data):
    """
    Calculates the SimHash value for the string property passed in.

    :param data: Contains the keyword arguments passed in. Requires a keyword "property",
    which is a string.
    :return: An integer value
    """

    return int(simhash.Simhash(data["property"]).value/100)


def hash_labels_prop(**data):
    """
    Computes a hash value using SimHash to hash all the labels on the node, and
    concatenates that value with the SimHash of the string property provided.

    :param data: Contains the keyword arguments passed in. Requires 3 keyword arguments:
    labels: A list of strings

    node_label_hash: A Dictionary of String -> integer
    Each label is assigned an integer which is a power of 2, to ensure that each combination
    of labels has a unique hash value.

    property: A string to hash on
    :return: An integer
    """

    node_label_hash = data["node_label_hash"]
    labels = data["labels"]
    node_property = data["property"]

    hash_value = hash_labels_only(labels=labels, node_label_hash=node_label_hash)
    hash_value *= 1e10
    hash_value += int(str(hash_simhash(property=node_property))[:10])
    return hash_value


def hash_labels_only(**data):
    """
    Computes a hash value only based on a list of labels.

    :param data: Contains the keyword arguments passed in. Requires 2 keyword arguments:
    labels: A list of strings

    node_label_hash: A Dictionary of String -> integer
    Each label is assigned an integer which is a power of 2, to ensure that each combination
    of labels has a unique hash value.
    :return: An integer
    """

    labels = data["labels"]
    node_label_hash = data["node_label_hash"]

    hash_value = 0
    for label in labels:
        hash_value += node_label_hash[label]

    return hash_value
