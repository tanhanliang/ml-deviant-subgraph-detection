"""
This file contains all optimisable parameters for the convolutional neural network.
"""
from patchy_san.neighborhood_assembly import get_ts
from patchy_san.graph_normalisation import build_node_list_hashing

# w
FIELD_COUNT = 2

# k
MAX_FIELD_SIZE = 5

# s
STRIDE = 2

# input channels
HASH_PROPERTIES = ['cmdline', 'name', 'ips', 'client_port', 'meta_login']

#a_v
CHANNEL_COUNT = len(HASH_PROPERTIES)

# Used to compute the hash value of a node. The hash values are base 2 to ensure that
# every unique combination of node labels will produce unique hash values.
# The hash value of the node will be computed partially from the addition of node's label values
NODE_TYPE_HASH = {'Conn': 2, 'File': 4, 'Global': 8, 'Machine': 16, 'Meta': 32, 'Process': 64,
                  'Socket': 1}

# Number of digits that can represent the range of values possible for each property
PROPERTY_CARDINALITY = {'cmdline': int(1e19), 'name': int(1e19), 'ips': int(1e10),
                        'client_port': int(1e5), 'meta_login': int(1e10)}

# A hash function used to canonicalise the graph (ie. represent the graph in such a way that
# isomorphic graphs have the same representation)
HASH_FN = hash

# A function used to impose an order on the nodes of a graph (ie. to linearise the nodes)
LABELING_FN = get_ts

# A function used to build the normalised node list for each receptive field
# Takes a Dictionary of node_id -> node as input, returns list of nodes
NORM_FIELD_FN = build_node_list_hashing
