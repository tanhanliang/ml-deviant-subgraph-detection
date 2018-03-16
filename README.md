# ml-deviant-subgraph-detection
This project aims to create a classifier which will identify subgraphs from a larger provenance graph which potentially contains malicious behavior.

### Getting stored data or data from pre-built rules
You don't have to have a Neo4j database of provenance data anymore! Simply fetch the stored training data as follows:
```
from patchy_san.cnn import build_model
from utility.load_data import load_data

# inputs is a list of 3 tensors: x_patchy_nodes, x_patchy_edges and x_embed
inputs, y = load_data()
```
This is how to load some stored data manually. The training data shapes can be found in
`make_training_data/about_training_data.txt`.
```
import numpy as np
x_patchy_nodes = np.fromfile("training_data/x_patchy_nodes6.txt")
x_patchy_edges = np.fromfile("training_data/x_patchy_edges6.txt")
x_embed = np.fromfile("training_data/x_embed6.txt")
y = np.fromfile("training_data/y_train6.txt")

x_patchy_nodes = x_patchy_nodes.reshape((2000, 6, 5, 1))
x_patchy_edges = xpe.reshape((2000, 36, 2, 1))
x_embed = xe.reshape((2000, 120))
y = y.reshape((2000, 2))
```

If you do have a Neo4j database of provenance data, you can build some training data as follows:
```
from make_training_data.format_training_data import get_final_datasets
from make_training_data.fetch_training_data import get_train_6_node_general

results = get_train_6_node_general()
x_patchy_nodes, x_patchy_edges, x_embed, y = get_final_datasets(results)
```

This is how to synthesise some training data:
```
from make_training_data.synthesise_training_data import get_graphs_test_negative_data
from make_training_data.format_training_data import process_training_examples

training_graphs = get_graphs_test_negative_data()
xpn,xpe,xe,y = process_training_examples(training_graphs)
```

### Making your own training data

1. Add a function in ```make_training_data/fetch_training_data.py``` with the Neo4j query used to
fetch the training data you want.
2. (Optional) Add a function in ```make_training_data/synthesise_training_data.py``` to tweak your data.
3. Process the results (BoltStatementResults) or training_graphs, build and then train the model!

If you chose to not alter the queried data:
```
from make_training_data.fetch_training_data import # Your function here
from make_training_data.format_training_data import get_final_datasets

results = # Your function here
xpn,xpe,xe,y = get_final_datasets(results)
```

If you chose to alter the queried data:
```
from make_training_data.synthesise_training_data import # Your function here
from make_training_data.format_training_data import process_training_examples

training_graphs = # Your function here
xpn,xpe,xe,y = process_training_examples(training_graphs)
```

If you want to change the parameters of the model, such as receptive field size, receptive
field count, number of classes, see `patchy_san/parameters.py`.

### Training the model
`xpn` is the Patchy-San input for nodes, `xpe` is the Patchy-San input for edges, `xe` is the embedding
input for node properties and `y` is the target output values.
```
from patchy_san.cnn import build_model()
model = build_model()
model.fit([xpn,xpe,xe], y, validation_split=0.20, epochs=10, batch_size=10)
```
Yay!

The project dependencies may be found in requirements.txt.

Phases of the project:
1) Exploring training data (neo4j graphs), determine how best to model provenance data. The current candidate model is a convolutional neural network based on the Patchy-San algorithm.

2) Determining how to extract data from model of process execution that can be used by a Machine Learning classifier. Rules will be developed by hand to create the training data to train the CNN with.

3) Building and training classifiers.

4) Evaluation of methods used.

Currently, this project is in phase 4.
