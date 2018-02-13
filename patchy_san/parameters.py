"""
This file contains all optimisable parameters for the convolutional neural network.
"""
from optimisable_functions.labeling_fns import get_ts
from optimisable_functions.hashes import hash_simhash, hash_labels_prop, hash_labels_only

# w
FIELD_COUNT = 1

# k
MAX_FIELD_SIZE = 3

# s
STRIDE = 3

# input channels
HASH_PROPERTIES = ['cmdline', 'name', 'ips', 'client_port', 'meta_login']
# HASH_PROPERTIES = ['cmdline', 'name']


#a_v
CHANNEL_COUNT = len(HASH_PROPERTIES)

# Number of digits that can represent the range of values possible for each property
PROPERTY_CARDINALITY = {'cmdline': int(1e19), 'name': int(1e19), 'ips': int(1e10),
                        'client_port': int(1e5), 'meta_login': int(1e10)}

# A hash function used to canonicalise the graph (ie. represent the graph in such a way that
# isomorphic graphs have the same representation)
# HASH_FN = hash_simhash
HASH_FN = hash_labels_only

# A function used to impose an order on the nodes of a graph (ie. to linearise the nodes)
LABELING_FN = get_ts

# A function used to build the normalised node list for each receptive field
# Takes a Dictionary of node_id -> node as input, returns list of nodes

# NORM_FIELD_FN is now imported in patchy_san.make_cnn_input
# NORM_FIELD_FN = build_node_list_hashing

# Number of classes that model should predict
CLASS_COUNT = 2

DEFAULT_TENSOR_VAL = 0
