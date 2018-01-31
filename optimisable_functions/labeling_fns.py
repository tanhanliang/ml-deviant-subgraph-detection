"""
Contains functions to compute a value for each node to allow nodes in a graph to be arranged in a
linear order.
"""


def get_ts(node):
    """
    Given a node, returns its timestamp if it exists, otherwise throws a RuntimeError.
    This fn will be used to sort a list of nodes by timestamp, using the built in sorted()
    function.

    :param node: A node in a list to be sorted
    :return: The timestamp of the node
    """

    if 'timestamp' not in node.properties:
        raise RuntimeError('timestamp does not exist in properties dict of node')

    return node.properties['timestamp']

