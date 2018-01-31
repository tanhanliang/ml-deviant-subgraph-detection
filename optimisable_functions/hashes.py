"""This file contains functions to apply different hash functions to strings"""

import simhash


def hash_simhash(text):
    """
    Hashes the value using the SimHash algorithm.

    :param text: The string value to be hashed.
    :return: A hash value as an integer
    """

    return int(simhash.Simhash(text).value/100)
