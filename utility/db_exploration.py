"""
This file contains functions to help me explore the Neo4j database.
"""

from neo4j.v1 import GraphDatabase, basic_auth


def get_attack_nodes():
    """
    Searches the database for all nodes with 'attack' in their cmdline.

    :return: A list of nodes
    """

    driver = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "neo4j"))
    session = driver.session()

    query = """
    MATCH (n) WHERE n.cmdline =~ '.*attack.*' RETURN n
    """
    results = session.run(query)
    session.close()

    attack_nodes = []
    for result in results:
        attack_nodes.append(result['n'])

    return attack_nodes


def get_attack_paths(attack_nodes):
    """
    Given a list of nodes from which an attack is launched, returns a list of sets of paths
    accessed by each attack node.

    :param attack_nodes: A list of nodes
    :return: A list of sets of paths (String)
    """

    driver = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "neo4j"))
    session = driver.session()

    path_sets = []
    for node in attack_nodes:
        query = """
        MATCH (n)-[r*]->(m) 
        WHERE Id(m) = $node_id
        RETURN n
        """
        results = session.run(query, {"node_id": node.id})
        paths = set()

        for result in results:
            if 'name' in result['n'].properties and len(result['n'].properties['name']) > 0:
                paths.add(result['n'].properties['name'][0])

        path_sets.append(paths)
    session.close()
    return path_sets
