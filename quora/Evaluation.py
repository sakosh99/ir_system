import pandas as pd
import os
from QuoraServer import run_query_with_index, run_query_with_cluster, clean_docs, clean_queries

qrels = pd.read_csv(os.path.join(os.getcwd(), "resources", "original_qrels.csv"))
qrels = qrels.fillna('')


def run_queries(queries, cluster=False):
    precision_at_10_list = []
    recall_list = []
    precision_list = []
    average_precision_list = []
    mrr_list = []
    f_measure_list = []

    clean_docs_len = len(clean_docs)

    for query in queries:
        relevant_docs_idx = qrels.query(f'query_id == {query[1]}')['doc_id'].values
        relevant_docs_count = len(relevant_docs_idx)

        not_relevant_docs_count = clean_docs_len - relevant_docs_count

        if cluster:
            result = run_query_with_cluster([query[2]])[:relevant_docs_count]
        else:
            result = run_query_with_index([query[2]])[:relevant_docs_count]

        retrieved_docs_idx = [element['doc_id'] for element in result]

        retrieved_docs_count = len(result)

        not_retrieved_docs_count = clean_docs_len - retrieved_docs_count

        relevant_retrieved_at_k_count = get_relevant_retrieved_at_k_count(relevant_docs_idx, retrieved_docs_idx, 10)
        relevant_retrieved_count = get_relevant_retrieved_count(relevant_docs_idx, retrieved_docs_idx)

        precision_at_10 = get_precision(relevant_retrieved_at_k_count, 10)
        precision_at_10_list.append(precision_at_10)

        precision = get_precision(relevant_retrieved_count, retrieved_docs_count)
        precision_list.append(precision)

        recall = get_recall(relevant_retrieved_count, relevant_docs_count)
        recall_list.append(recall)

        average_precision = get_average_precision(relevant_docs_idx, retrieved_docs_idx)
        average_precision_list.append(average_precision)

        mrr = get_mrr(relevant_docs_idx, retrieved_docs_idx)
        mrr_list.append(mrr)

        f_measure = get_f_measure(precision, recall)
        f_measure_list.append(f_measure)

    avg_precision_at_10 = sum(precision_at_10_list) / len(precision_at_10_list)
    avg_precision = sum(precision_list) / len(precision_list)
    avg_recall = sum(recall_list) / len(recall_list)
    avg_map = sum(average_precision_list) / len(average_precision_list)
    avg_mrr = sum(mrr_list) / len(mrr_list)
    mean_f_measure = sum(f_measure_list) / len(f_measure_list)

    print("############################################")
    print(f"Average Precision@10: {avg_precision_at_10}")
    print(f"Average Recall: {avg_recall}")
    print(f"Average Precision: {avg_precision}")
    print(f"Mean Average Precision (MAP): {avg_map}")
    print(f"Average MRR: {avg_mrr}")
    print(f"Mean F Measure: {mean_f_measure}")


def get_relevant_retrieved_at_k_count(relevant_docs_idx, retrieved_docs_idx, k):
    retrieved_docs_idx_at_k = retrieved_docs_idx[:k]
    relevant_retrieved_at_k_idx = set(relevant_docs_idx) & set(retrieved_docs_idx_at_k)
    return len(relevant_retrieved_at_k_idx)


def get_relevant_retrieved_count(relevant_docs_idx, retrieved_docs_idx):
    relevant_retrieved_idx = set(relevant_docs_idx) & set(retrieved_docs_idx)
    return len(relevant_retrieved_idx)


# Used for precision and for precision@K
def get_precision(relevant_retrieved_count, retrieved_count):
    if retrieved_count == 0:
        return 0
    return relevant_retrieved_count / retrieved_count


def get_recall(relevant_retrieved_count, relevant_count):
    if relevant_count == 0:
        return 0
    return relevant_retrieved_count / relevant_count


def get_average_precision(relevant_docs, retrieved_docs):
    average_precision = 0.0
    relevant_docs_count = len(relevant_docs)
    correct_count = 0
    precision_at_rank = []

    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            correct_count += 1
            precision = correct_count / (i + 1)
            precision_at_rank.append(precision)

    if correct_count > 0:
        average_precision = sum(precision_at_rank) / relevant_docs_count

    return average_precision


def get_mrr(relevant_docs, retrieved_docs):
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            return 1 / (i + 1)
    return 0


def get_f_measure(precision, recall):
    if precision > 0 and recall > 0:
        return (2 * precision * recall) / (precision + recall)
    return 0


if __name__ == "__main__":
    run_queries(clean_queries.values[:10], True)
