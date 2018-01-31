# ml-deviant-subgraph-detection
This project aims to create a classifier which will identify subgraphs from a larger provenance graph which potentially contains malicious behavior.

To extract all training data and train the model:  

```
from patchy_san.cnn import build_model
from make_training_data.format_training_data import get_final_datasets
model = build_model()
x, y = get_final_datasets()
model.fit(x, y, validation_split=0.33, epochs=150, batch_size=5)
```

This assumes that you have a neo4j database containing provenance data collected from the CADETS/OPUS projects. This also
assumes that you have TensorFlow and Keras installed.

Phases of the project:
1) Exploring training data (neo4j graphs), determine how best to model provenance data. The current candidate model is a convolutional neural network based on the Patchy-San algorithm.

2) Determining how to extract data from model of process execution that can be used by a Machine Learning classifier. Rules will be developed by hand to create the training data to train the CNN with.

3) Building and training classifiers.

4) Writing code to visualise subgraphs on the cadets-ui.

5) Evaluation of methods used.

Currently, this project is in phase 3.
