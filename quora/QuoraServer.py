import os
import pickle
import re

import numpy as np
import pandas as pd
import spacy
from autocorrect import Speller
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from DataProcessing import date_processing, lemmatize_text, abbreviations_processing

clean_docs = pd.read_csv(os.path.join(os.getcwd(), "resources", "clean_docs.csv"))
clean_docs = clean_docs.fillna('')

original_docs = pd.read_csv(os.path.join(os.getcwd(), "resources", "original_docs.csv"))
original_docs = original_docs.fillna('')

clean_queries = pd.read_csv(os.path.join(os.getcwd(), "resources", "clean_queries.csv"))
clean_queries = clean_queries.fillna('')

original_queries = pd.read_csv(os.path.join(os.getcwd(), "resources", "original_queries.csv"))
original_queries = original_queries.fillna('')

docs_index = pickle.load(open(os.path.join(os.getcwd(), "resources", "docs_index.pkl"), 'rb'))

query_index = pickle.load(open(os.path.join(os.getcwd(), "resources", "query_index.pkl"), 'rb'))


def get_relevant_idx(query, index_array):
    relevant_idx = set()
    for word in query.split():
        if word in index_array:
            for idx in index_array[word]:
                relevant_idx.add(idx)
    return relevant_idx


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
vectorizer = TfidfVectorizer(stop_words='english')
spell_correction = Speller(lang='en')


def query_processing(query):
    # If Query Contains Numbers Only
    text = str(query)

    # Date Processing
    text = date_processing(text)

    # Abbreviations Processing
    text = abbreviations_processing(text)

    # To lower case
    text = text.lower()

    # Remove Punctuation
    text = re.sub(r'[^_\w\s]', '', text)

    # Lemmatize text using spaCy
    if text != '':
        text = lemmatize_text(text)

    # Remove non-alphanumeric Characters
    text = re.compile('[^_a-zA-Z0-9\s]').sub('', str(text))

    # Spell Correction
    text = spell_correction(text)

    return text


kmeans = pickle.load(open(os.path.join(os.getcwd(), "resources", "kmeans_model.pkl"), 'rb'))
svd = pickle.load(open(os.path.join(os.getcwd(), "resources", "svd.pkl"), 'rb'))
vectorized_docs = pickle.load(open(os.path.join(os.getcwd(), "resources", "vectorized_docs.pkl"), 'rb'))
cluster_vectorizer = pickle.load(open(os.path.join(os.getcwd(), "resources", "cluster_vectorizer.pkl"), 'rb'))


def run_query_with_index(query, evaluation=False, page=None, score_rate=0):
    if evaluation:
        processed_query = query_processing(query)
        print("Processed Query: " + str(processed_query))
    else:
        processed_query = query

    related_docs_idx = list(get_relevant_idx(processed_query, docs_index))
    if len(related_docs_idx) > 0:
        vectorized_docs = vectorizer.fit_transform(clean_docs['text'][related_docs_idx])

        vectorized_query = vectorizer.transform([processed_query])

        sorted_results = run_cosine_similarity(vectorized_query, vectorized_docs)

        last = []
        result_list = sorted_results
        if page is not None:
            result_list = result_list[(int(page) * 10):((int(page) + 1) * 10)]
        for res in result_list:
            if res[1] > score_rate:
                last.append({
                    'score': res[1],
                    'doc_id': original_docs['doc_id'][related_docs_idx[res[0]]],
                    'text': original_docs['text'][related_docs_idx[res[0]]]
                })
        if evaluation:
            return last
        else:
            return last, len(sorted_results)
    else:
        if evaluation:
            return []
        else:
            return [], 0


def get_suggestions(processed_query):
    related_queries_idx = list(get_relevant_idx(processed_query, query_index))

    if len(related_queries_idx) > 0:
        vectorized_docs = vectorizer.fit_transform(clean_queries['text'][related_queries_idx])

        vectorized_query = vectorizer.transform([processed_query])

        sorted_results = run_cosine_similarity(vectorized_query, vectorized_docs)

        last = []
        for res in sorted_results:
            last.append({
                'score': res[1],
                'doc_id': original_queries['query_id'][related_queries_idx[res[0]]],
                'text': original_queries['text'][related_queries_idx[res[0]]]
            })
        return last[:5]
    else:
        return []


def run_query_with_cluster(query, evaluation=False, page=None, score_rate=0):
    if evaluation:
        processed_query = query_processing(query)
        print("Processed Query: " + str(processed_query))
    else:
        processed_query = query

    query_vector = cluster_vectorizer.transform([processed_query])


    query_vector_svd = svd.transform(query_vector)


    nearest_cluster = kmeans.predict(query_vector_svd)[0]


    cluster_indices = np.where(kmeans.labels_ == nearest_cluster)[0]
    cluster_documents = clean_docs.iloc[cluster_indices]

    sorted_results = run_cosine_similarity(query_vector, vectorized_docs[cluster_indices], True, cluster_documents)

    last = []
    result_list = sorted_results
    if page is not None:
        result_list = result_list[(int(page) * 10):((int(page) + 1) * 10)]
    for _, row in result_list.iterrows():
        if row['score'] > score_rate:
            last.append({
                'score': row['score'],
                'doc_id': row['doc_id'],
                'text': original_docs['text'][[row['id']]].tolist()[0]
            })
    if evaluation:
        return last
    else:
        return last, len(sorted_results)


def run_cosine_similarity(vectorized_query, vectorized_text_list, clustering=False, cluster_documents=None):
    similarity_scores = cosine_similarity(vectorized_query, vectorized_text_list)

    if not clustering:
        results = list(enumerate(similarity_scores[0]))

        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

        return sorted_results

    else:
        cluster_documents_copy = cluster_documents.copy()
        cluster_documents_copy['score'] = similarity_scores.flatten()

        sorted_results = cluster_documents_copy.sort_values('score', ascending=False)

        return sorted_results
