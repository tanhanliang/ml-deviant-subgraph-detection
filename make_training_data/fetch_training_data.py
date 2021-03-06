"""
This module contains functions to fetch data from the database according to particular rules
(e.g. process downloaded a file from the internet, then executed it).
Each function corresponds to one rule.
"""
from data_processing.neo4j_interface_fns import execute_query


def get_train_3_node_simple():
    """
    Gets training data for 3-node patterns.
    First pattern: process connects to a socket, writes to a file.
    Second pattern: process connects to a socket, connects to another process.

    :return: A list of BoltStatementResult objects
    """

    download_file_write = """
    MATCH path1=(process:Process)<-[write]-(:File)
    MATCH path2=(process:Process)<-[]-(:Socket)
    WHERE (write.state = 'RaW' OR write.state = 'WRITE')
    RETURN path1, path2 LIMIT 2000
    """

    proc_proc_soc = """
    MATCH p1=(process:Process)<-[]-(sock:Socket)
    MATCH p2=(process:Process)<-[]-(proc:Process)
    RETURN p1,p2 LIMIT 2000
    """

    negative_data = """
    MATCH path1=(n1)<-[r1]-(m1)
    MATCH path2=(n1)<-[r2]-(m2)
    WHERE m1 <> m2 AND (NOT "File" IN labels(m1) OR NOT "Socket" IN labels(m2)) 
    AND (NOT "Process" IN labels(n1)) AND (r1.state <> 'RaW' AND r1.state <> 'WRITE')
    RETURN path1, path2 LIMIT 2000
    """

    return [execute_query(download_file_write),
            execute_query(proc_proc_soc),
            execute_query(negative_data)]


def get_train_4_node_diff_name():
    """
    Gets training data where the pattern of nodes in the graph is the same: A process
    connects to a socket, reads a file and executes another file. The only difference
    is in the name of the file read.

    Pattern 1: Name is '/etc/libmap.conf'
    Pattern 2: Name is '/lib/libc.so.7'
    Pattern 3: Name is '/lib/libcrypto.so.8'

    :return: A list of BoltStatementResult objects
    """

    general_pattern = """
    MATCH path1=(process:Process)<-[conn]-(sock:Socket)
    MATCH path2=(process:Process)<-[read]-(file:File)
    MATCH path3=(process:Process)<-[exec]-(file2:File)
    WHERE read.state="READ" AND exec.state="BIN" AND file2.name[0] =~ '/usr.*' AND file <> file2
    AND file.name[0] = '%s'
    RETURN path1,path2,path3 LIMIT 1000
    """

    negative_data = """
    MATCH path1=(node1)<-[r1]-(node2)
    MATCH path2=(node1)<-[r2]-(node3)
    MATCH path3=(node1)<-[r3]-(node4)
    WHERE (NOT "Process" in labels(node1) OR NOT "Socket" IN labels(node2) OR NOT "File" IN labels(node3)
    OR NOT "File" IN labels(node4) OR r2.state <> "READ" OR r3.state <> "BIN"
    OR NOT node4.name[0] =~ '/usr.*' OR NOT (node3.name[0] = '/etc/libmap.conf'
    OR node3.name[0] = '/lib/libc.so.7' OR node4.name[0] = '/lib/libcrypto.so.8')) AND node3 <> node4
    
    RETURN path1,path2,path3 LIMIT 1000
    """

    names = ['/etc/libmap.conf', '/lib/libc.so.7', '/lib/libcrypto.so.8']

    results = []
    for name in names:
        results.append(execute_query(general_pattern % name))

    results.append(execute_query(negative_data))
    return results


def get_train_4_node_test_cmdline():
    """
    Creates training data where the only constant difference between the pattern and the
    negative data is the presence of the word '-k' somewhere in the cmdline of a particlar node.

    :return: A list of BoltStatementResults results
    """

    pattern = """
    MATCH path1=(node1)<-[]-(node2)
    MATCH path2=(node1)<-[]-(node3)
    MATCH path3=(node1)<-[]-(node4)
    WHERE node2 <> node3 AND node3 <> node4 AND node2 <> node4 AND node1.cmdline =~ '.*-k.*'
    RETURN path1,path2,path3 LIMIT 1000
    """

    negative_data = """
    MATCH path1=(node1)<-[]-(node2)
    MATCH path2=(node1)<-[]-(node3)
    MATCH path3=(node1)<-[]-(node4)
    WHERE node2 <> node3 AND node3 <> node4 AND node2 <> node4 AND NOT node1.cmdline =~ '.*-k.*'
    RETURN path1,path2,path3 LIMIT 1000
    """

    return [execute_query(pattern), execute_query(negative_data)]


def get_train_4_node_simple():
    """
    Queries the db twice for the same simple 4 node pattern. This is used to help me
    synthesise training data (it is queried twice so the pipeline will partition the data
    by assigning different labels, so I can edit data in one half of it).

    :return: A list of BoltStatementResult objects
    """

    pattern = """
    MATCH path1=(node1)<-[]-(node2),
    path2=(node1)<-[]-(node3),
    path3=(node1)<-[]-(node4)
    RETURN path1,path2,path3 LIMIT 1000
    """

    return [execute_query(pattern), execute_query(pattern)]


def get_train_6_node_general():
    """
    Queries the database for a particular 6-node pattern.

    :return: A list of BoltStatementResult objects.
    """

    pattern = """
    MATCH path1=(node1)<-[r1]-(node2),
    path2=(node1)<-[r2]-(node3),
    path3=(node1)<-[r3]-(node4),
    path4=(node3)<-[r4]-(node5),
    path5=(node3)<-[r5]-(node6)
    RETURN path1,path2,path3,path4,path5 LIMIT 1000
    """

    return [execute_query(pattern), execute_query(pattern)]


def get_train_8_nodes_general():

    pattern = """
    MATCH path1=(node1)<-[]-(node2),
    path2=(node1)<-[]-(node3),
    path3=(node1)<-[]-(node4),
    path4=(node3)<-[]-(node5),
    path5=(node3)<-[]-(node6),
    path6=(node5)<-[]-(node7),
    path7=(node6)<-[]-(node8)
    RETURN path1,path2,path3,path4,path5,path6,path7 LIMIT 1000
    """

    return [execute_query(pattern), execute_query(pattern)]


def get_train_10_nodes_general():

    pattern = """
    MATCH path1=(node1)<-[]-(node2),
    path2=(node1)<-[]-(node3),
    path3=(node1)<-[]-(node4),
    path4=(node3)<-[]-(node5),
    path5=(node3)<-[]-(node6),
    path6=(node5)<-[]-(node7),
    path7=(node6)<-[]-(node8),
    path8=(node9)<-[]-(node6),
    path9=(node10)<-[]-(node9)
    RETURN path1,path2,path3,path4,path5,path6,path7,path8,path9 LIMIT 1000
    """

    return [execute_query(pattern), execute_query(pattern)]


def get_train_12_nodes_general():

    pattern = """
    MATCH path1=(node1)<-[]-(node2),
    path2=(node1)<-[]-(node3),
    path3=(node1)<-[]-(node4),
    path4=(node3)<-[]-(node5),
    path5=(node3)<-[]-(node6),
    path6=(node5)<-[]-(node7),
    path7=(node6)<-[]-(node8),
    path8=(node9)<-[]-(node6),
    path9=(node10)<-[]-(node9),
    path10=(node11)<-[]-(node2),
    path11=(node12)<-[]-(node4)
    RETURN path1,path2,path3,path4,path5,path6,path7,path8,path9,path10,path11 LIMIT 1000
    """

    return [execute_query(pattern), execute_query(pattern)]


def get_train_16_nodes_general():

    pattern = """
    MATCH path1=(node1)<-[]-(node2),
    path2=(node1)<-[]-(node3),
    path3=(node1)<-[]-(node4),
    path4=(node3)<-[]-(node5),
    path5=(node3)<-[]-(node6),
    path6=(node5)<-[]-(node7),
    path7=(node6)<-[]-(node8),
    path8=(node9)<-[]-(node6),
    path9=(node10)<-[]-(node9),
    path10=(node6)<-[]-(node11),
    path11=(node6)<-[]-(node12),
    path12=(node6)<-[]-(node13),
    path13=(node6)<-[]-(node14)
    RETURN path1,path2,path3,path4,path5,path6,path7,path8,path9,path10,path11,path12,path13 LIMIT 1000
    """

    return [execute_query(pattern), execute_query(pattern)]