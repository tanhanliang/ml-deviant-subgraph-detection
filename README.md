# ml-deviant-subgraph-detection
This project aims to create a classifier which will identify subgraphs from a larger provenance graph which potentially contains malicious behavior.

You don't have to have a Neo4j database of provenance data anymore! Simply fetch the stored training data as follows:
```
from patchy_san.cnn import build_model
from utility.load_data import *
x_patchy, x_embed, y = load_data()
model = build_model()
model.fit([x_patchy, x_embed], y, validation_split=0.20, epochs=10, batch_size=5)
```

If you do have a Neo4j database of provenance data, you can build training data as follows:
```
from make_training_data.format_training_data import get_final_datasets
# An example of training data that you can build
from make_training_data.fetch_training_data import get_train_4_node_test_cmdline
results = get_train_4_node_test_cmdline()
x_patchy, x_embed, y = get_final_datasets(results)
```

The project dependencies may be found in requirements.txt.

Phases of the project:
1) Exploring training data (neo4j graphs), determine how best to model provenance data. The current candidate model is a convolutional neural network based on the Patchy-San algorithm.

2) Determining how to extract data from model of process execution that can be used by a Machine Learning classifier. Rules will be developed by hand to create the training data to train the CNN with.

3) Building and training classifiers.

4) Evaluation of methods used.

Currently, this project is in phase 3.
