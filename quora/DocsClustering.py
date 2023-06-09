import os
import pickle

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

clean_docs = pd.read_csv(os.path.join(os.getcwd(), "resources", "clean_docs.csv"))
clean_docs = clean_docs.fillna('')

vectorizer = TfidfVectorizer(stop_words='english')


def document_clustering():
    vectorized_docs = vectorizer.fit_transform(clean_docs['text'])

    svd = TruncatedSVD(n_components=100)
    vectorized_docs_svd = svd.fit_transform(vectorized_docs)
    kmeans = KMeans(n_clusters=7, random_state=0, n_init=1)
    kmeans.fit(vectorized_docs_svd)

    pickle.dump(vectorizer, open(os.path.join(os.getcwd(), "resources", "cluster_vectorizer.pkl"), 'wb'))
    pickle.dump(svd, open(os.path.join(os.getcwd(), "resources", "svd.pkl"), 'wb'))
    pickle.dump(vectorized_docs, open(os.path.join(os.getcwd(), "resources", "vectorized_docs.pkl"), 'wb'))
    pickle.dump(kmeans, open(os.path.join(os.getcwd(), "resources", "kmeans_model.pkl"), 'wb'))
