# ml-deviant-subgraph-detection
This project aims to create a classifier which will identify subgraphs from a larger provenance graph which potentially contains malicious behavior.

Phases of the project:
1) Exploring training data (neo4j graphs), determine how best to model provenance data. Current candidate models are linearizing events and clustering based on edge betweenness measures.

2) Determining how to extract data from model of process execution that can be used by a Machine Learning classifier. Also deciding which classifier is most appropriate.

3) Building algorithms to model process execution.

4) Building classifiers. 

5) Writing code to visualise subgraphs on the cadets-ui.

6) Evaluation of methods used.

Currently, this project is in phase 1.
