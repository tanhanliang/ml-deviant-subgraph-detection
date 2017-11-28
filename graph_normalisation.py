"""
Contains functions to normalise graph nodes in a linear ordering such that similar graphs
have nodes ordered similarly (relative ordering of nodes) after being normalised.
"""
from neo4j_interface_fns import *

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

