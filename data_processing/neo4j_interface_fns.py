"""
This module contains functions to extract data from a neo4j database.

Dependencies: neo4j-driver
"""

from neo4j.v1 import GraphDatabase, basic_auth


def get_subgraph_paths(root_id, end_id):
    """
    Queries the neo4j database for all paths starting from a root node and ending
    at an end node

    :param root_id: The database Id of the root node
    :param end_id: The database Id of the end node
    :return: A BoltStatementResult object describing all paths between root node
    specified by root_id, and end node specified by end_id.
    """
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "neo4j"))
    session = driver.session()

    query = """
    MATCH path=(n)-[r*]->(m)
    WHERE Id(n) = $id1 AND Id(m) = $id2
    RETURN path
    """
    results = session.run(query, {"id1": root_id, "id2": end_id})
    session.close()
    return results


def get_neighborhood(start_id, k):
    """
    Returns all nodes and edges in a neighborhood spawned by a certain node, for example
    all nodes with edges pointing to the start node, all nodes with edges pointing to these
    nodes until neighborhood has size k. Similar idea to the bacon number.

    :param start_id: The Id of start node
    :param k: Neighborhood size
    :return: A BoltStatementResult object describing all paths in the neighborhood
    """
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "neo4j"))
    session = driver.session()

    query = """
    MATCH path=(n)-[r*]->(m) WHERE
    Id(m) = $id
    RETURN path
    LIMIT $num
    """
    results = session.run(query, {"id": start_id, "num": k})
    session.close()
    return results


def execute_query(query):
    """
    Executes a given query.

    :param query: A String representing the query to be executed
    :return: A BoltStatementResult object
    """

    driver = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "neo4j"))
    session = driver.session()

    results = session.run(query)
    session.close()
    return results
