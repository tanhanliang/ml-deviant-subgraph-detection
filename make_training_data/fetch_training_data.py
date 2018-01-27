"""
This module contains functions to fetch data from the database according to particular rules
(e.g. process downloaded a file from the internet, then executed it).
Each function corresponds to one rule.
"""
from data_processing.neo4j_interface_fns import execute_query


DOWNLOAD_FILE_WRITE = """
MATCH path1=(n1)<-[r1]-(:File)
MATCH path2=(n1)<-[]-(:Socket)
WHERE (r1.state = 'RaW' OR r1.state = 'WRITE')
RETURN path1, path2
"""

TRIPLE_NODES = """
MATCH path1=(n1)-[r1]-(m1)
MATCH path2=(n1)-[r2]-(m2)
WHERE m1 <> m2
RETURN path1, path2 LIMIT 10000
"""


def get_train_download_file_execute():
    """
    Gets training data for instances of a process downloading a file then executing it.

    :return: A BoltStatementResult object
    """

    return execute_query(DOWNLOAD_FILE_WRITE)


def get_train_all_triples():
    """
    Gets all triple nodes with the following configuration: (node1)->(node2)<-(node3).

    :return: A BoltStatementResult object
    """

    return execute_query(TRIPLE_NODES)
