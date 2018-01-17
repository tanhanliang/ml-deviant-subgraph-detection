"""
This module contains functions to fetch data from the database according to particular rules
(e.g. process downloaded a file from the internet, then executed it).
Each function corresponds to one rule.
"""
from data_processing.neo4j_interface_fns import execute_query


DOWNLOAD_FILE_EXECUTE = """
MATCH path1=(n1)<-[r1]-(:File)
WHERE (r1.state = 'RaW' OR r1.state = 'WRITE')
WITH n1,path1
MATCH path2=(n1)<-[]-(:Socket)
RETURN path1,path2
"""


def get_train_data_download_file_execute():
    """
    Gets training data for instances of a process downloading a file then executing it.
    The result is processed to produce nodes and edges.

    :return: A BoltStatementResult object
    """

    return execute_query(DOWNLOAD_FILE_EXECUTE)
