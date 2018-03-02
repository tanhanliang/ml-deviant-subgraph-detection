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
RETURN path1, path2 LIMIT 2000
"""

PROC_PROC_SOCK = """
MATCH p1=(process:Process)<-[]-(sock:Socket)
MATCH p2=(process:Process)<-[]-(proc:Process)
RETURN p1,p2 LIMIT 2000
"""

NEGATIVE_DATA_TRIPLES = """
MATCH path1=(n1)<-[r1]-(m1)
MATCH path2=(n1)<-[r2]-(m2)
WHERE m1 <> m2 AND (NOT "File" IN labels(m1) OR NOT "Socket" IN labels(m2)) 
AND (r1.state <> 'RaW' AND r1.state <> 'WRITE')

AND m1 <> m2 AND (NOT "Process" IN labels(m1) OR NOT "Socket" IN labels(m2)) 

RETURN path1, path2 LIMIT 2000
"""

DOWNLOAD_FILE_WRITE_EXECUTE = """
MATCH path1=(process:Process)<-[write]-(:File)
MATCH path2=(process:Process)<-[]-(:Socket)
MATCH path3=(process:Process)<-[bin]-(:File)
WHERE write.state = 'WRITE' AND bin.state = "BIN"
RETURN path1,path2,path3 LIMIT 1000
"""

READ_EXEC_CONN_LIBMAP = """
MATCH path1=(process:Process)<-[conn]-(sock:Socket)
MATCH path2=(process:Process)<-[read]-(file:File)
MATCH path3=(process:Process)<-[exec]-(file2:File)
WHERE read.state="READ" AND exec.state="BIN" AND file2.name[0] =~ '/usr.*' AND file <> file2
AND file.name[0] = '/etc/libmap.conf'
RETURN path1,path2,path3 LIMIT 1000
"""

READ_EXEC_CONN_LIBC1 = """
MATCH path1=(process:Process)<-[conn]-(sock:Socket)
MATCH path2=(process:Process)<-[read]-(file:File)
MATCH path3=(process:Process)<-[exec]-(file2:File)
WHERE read.state="READ" AND exec.state="BIN" AND file2.name[0] =~ '/usr.*' AND file <> file2
AND file.name[0] = '/lib/libc.so.7'
RETURN path1,path2,path3 LIMIT 1000
"""

READ_EXEC_CONN_LIBC2 = """
MATCH path1=(process:Process)<-[conn]-(sock:Socket)
MATCH path2=(process:Process)<-[read]-(file:File)
MATCH path3=(process:Process)<-[exec]-(file2:File)
WHERE read.state="READ" AND exec.state="BIN" AND file2.name[0] =~ '/usr.*' AND file <> file2
AND file.name[0] = '/lib/libcrypto.so.8'
RETURN path1,path2,path3 LIMIT 1000
"""

NEGATIVE_DATA_4_NODES = """
MATCH path1=(node1)<-[r1]-(node2)
MATCH path2=(node1)<-[r2]-(node3)
MATCH path3=(node1)<-[r3]-(node4)
WHERE (NOT "Process" in labels(node1) OR NOT "Socket" IN labels(node2) OR NOT "File" IN labels(node3)
OR NOT "File" IN labels(node4) OR r2.state <> "READ" OR r3.state <> "BIN" 
OR NOT node4.name[0] =~ '/usr.*' OR NOT (node3.name[0] = '/etc/libmap.conf' 
OR node3.name[0] = '/lib/libc.so.7' OR node4.name[0] = '/lib/libcrypto.so.8')) AND node3 <> node4

RETURN path1,path2,path3 LIMIT 1000
"""


def get_train_download_file_execute():
    """
    Gets training data for instances of a process downloading a file then executing it.

    :return: A BoltStatementResult object
    """

    return execute_query(DOWNLOAD_FILE_WRITE)


def get_train_proc_proc_socket():
    """
    Gets all triple nodes where a process connects to a socket and executes a file.

    :return: A BoltStatementResult object
    """

    return execute_query(PROC_PROC_SOCK)


def get_negative_data_triples():
    """
    Gets all triple nodes which does not match any pattern.

    :return: A BoltStatementResult object
    """

    return execute_query(NEGATIVE_DATA_TRIPLES)


def get_download_file_write_execute():
    """
    Gets all 4-node combinations where a process connects to a socket, downloads a file,
    and executes a file (probably not be the same file that was downloaded).

    :return: A BoltStatementResult object
    """

    return execute_query(DOWNLOAD_FILE_WRITE_EXECUTE)


def get_read_exec_conn():
    """
    Gets 4-node instances where a process connects to a socket, reads a file named '/etc/libmap.conf'
    and executes a file starting with '/usr'.

    :return: A BoltStatementResult object
    """

    return execute_query(READ_EXEC_CONN_LIBMAP)


def get_read_exec_conn_libc1():
    """
    Gets 4-node instances where a process connects to a socket, reads a file named '/lib/libc.so.7'
    and executes a file starting with '/usr'.

    :return: A BoltStatementResult object
    """

    return execute_query(READ_EXEC_CONN_LIBC1)


def get_read_exec_conn_libc2():
    """
    Gets 4-node instances where a process connects to a socket, reads a file named '/lib/libc.so.7'
    and executes a file starting with '/usr'.

    :return: A BoltStatementResult object
    """

    return execute_query(READ_EXEC_CONN_LIBC2)


def get_negative_data_4_nodes():
    """
    Gets all 4-node combinations which do not match any pattern.

    :return: A BoltStatementResult object
    """

    return execute_query(NEGATIVE_DATA_4_NODES)
