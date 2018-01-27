"""
This file contains functions to help me explore the Neo4j database.
"""

from neo4j.v1 import GraphDatabase, basic_auth
from data_processing.preprocessing import clean_data
# TODO: Put code creating and destroying sessions into separate optimisable_functions


def get_all_successor_nodes(root_id):
    """
    Returns all successor nodes given a root node.

    :param root_id: The id of the root node
    :return: A BoltStatementResult describing the raw result returned by Neo4j
    """

    driver = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "neo4j"))
    session = driver.session()

    query = """
    MATCH path=(n)-[r*]->(m) 
    WHERE Id(m) = $root_id
    RETURN path
    """
    results = session.run(query, {"root_id": root_id})
    session.close()
    return results


def get_attack_nodes():
    """
    Searches the database for all nodes with 'attack' in their cmdline.

    :return: A tuple of (nodes, edges). nodes is a Dictionary of node_id -> node, edges
    is a Dictionary of edge_id -> edge
    """

    driver = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "neo4j"))
    session = driver.session()

    query = """
    MATCH path=(n) WHERE n.cmdline =~ '.*attack.*' RETURN path
    """
    results = session.run(query)
    session.close()

    nodes, edges = clean_data(results)
    return nodes, edges


def get_attack_paths(nodes):
    """
    Given a list of nodes from which an attack is launched, returns a list of sets of paths
    accessed by each attack node.

    :param nodes: A list of nodes
    :return: A list of sets of paths (String)
    """

    driver = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "neo4j"))
    session = driver.session()

    path_sets = []
    for node_id in nodes:
        node = nodes[node_id]
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


def get_path_counts(path_sets):
    """
    Counts how many times each path is visited by all the attack nodes.

    :param path_sets: A list of sets of paths
    :return: A Dictionary of path (String) -> count
    """

    counts = {}

    for set in path_sets:
        for path in set:
            if path not in counts.keys():
                counts[path] = 0
            counts[path] += 1

    return counts


def print_paths_by_freq(counts):
    items = counts.items()
    items = sorted(items, key=lambda x: x[1])

    for item in items:
        print(item)


def print_properties(nodes, prop):
    """
    Prints the value of the property prop of each node that has that property.

    :param nodes: A Dictionary of node_id -> node
    :param prop: The property in question
    :return: nothing
    """

    for node_id, node in nodes.items():
        if prop in node.properties:
            print(node.properties[prop])
