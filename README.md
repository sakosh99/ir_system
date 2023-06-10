# IR System

This is an information retrieval (IR) system that is designed to cluster and indexing a collection of documents and evaluate the clustering and indexing performance using various metrics such as Precision, Recall, MAP, MRR, and so on.

## Getting Started

To use this IR system, you need to follow the steps below:


1. Run the `DataProcessing.py` file to preprocess the data and create the necessary files for clustering and indexing.
2. Run the `DocsClustering.py` file to build the clustering model.
3. Run the `indexer.py` file to build the index for documents and queries.
4. Run the REST APIs provided in the `QuoraRest.py` and `AntiqueRest.py`file to query the system and retrieve the results.

## Evaluation

To evaluate the performance of the system, you can use the `Evaluation.py` file, which computes various metrics such as Precision@10, Precision, Recall, MAP, MRR, and so on, based on the ground truth data.
