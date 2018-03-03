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
