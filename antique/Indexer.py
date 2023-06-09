import pandas as pd
import pickle
import os

clean_docs = pd.read_csv(os.path.join(os.getcwd(), "resources", "clean_docs.csv"))
clean_docs = clean_docs.fillna('')

clean_queries = pd.read_csv(os.path.join(os.getcwd(), "resources", "clean_queries.csv"))
clean_queries = clean_queries.fillna('')


def indexer(data_frame_texts):
    index = {}
    i = 0
    for text in data_frame_texts.values:
        for word in str(text[2]).split():
            if word not in index:
                index[word] = []
            index[word].append(i)
        i = i + 1
    return index


def index():
    docs_index = indexer(clean_docs)
    pickle.dump(docs_index, open(os.path.join(os.getcwd(), "resources", "docs_index.csv"), 'wb'))

    query_index = indexer(clean_queries)
    pickle.dump(query_index, open(os.path.join(os.getcwd(), "resources", "query_index.csv"), 'wb'))
