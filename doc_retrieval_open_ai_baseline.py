## This file is used to calculate the F1, Precision, Recall, MRR and MAP scores of the openAI GPT-3.5-turbo model against the bgb_eval dataset for the task of document retrieval
# NOTE: the embeddings of the model were already created up front and stores together with the datapoints in ./data/openAI/bgb_eval.json

import json
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def precision(actual_documents, expected_documents):
    """calculates the precision of the retrieved and expected docs"""
    intersection = len(set(actual_documents) & set(expected_documents))
    return intersection / len(actual_documents)

def recall(actual_documents, expected_documents):
    """calculates the recall of the retrieved and expected docs"""
    intersection = len(set(actual_documents) & set(expected_documents))
    return intersection / len(expected_documents)

def f1_score(actual_documents, expected_documents):
    """calculates the f1 score of the retrieved and expected docs"""
    p = precision(actual_documents, expected_documents)
    r = recall(actual_documents, expected_documents)
    if (p + r == 0):
        return 0
    return 2 * (p * r) / (p + r)

def mean_reciprocal_rank(retrieved_docs, relevant_docs):
    """calculates the mean reciprocal rank"""
    rr_list = []
    for actual, expected in zip(retrieved_docs, relevant_docs):
        for rank, doc_id in enumerate(actual, 1):
            if doc_id in expected:
                rr_list.append(1 / rank)
                break
        else:
            rr_list.append(0)
    return sum(rr_list) / len(rr_list)

def average_precision(retrieved_docs, relevant_docs):
    """calculates average precision"""
    ap_list = []
    for actual, expected in zip(retrieved_docs, relevant_docs):
        temp_precision = 0
        relevant_docs_count = 0
        for i, doc_id in enumerate(actual):
            if doc_id in expected:
                relevant_docs_count += 1
                temp_precision += relevant_docs_count / (i + 1)
        ap_list.append(temp_precision / len(expected))
    return sum(ap_list) / len(ap_list)

# load the document collections, which are the legal paragraphs of the BGB
doc_collection = []
with open("./data/openAI/bgb.json", "r") as f:
    doc_collection = json.load(f)

# load the questions, which also include the openai encodings
questions = []
with open("./data/openAI/bgb_eval.json", "r") as f:
    questions = json.load(f)

precisions = []
recalls = []
f1s = []
retrieved_docs = []
relevant_docs = []

# run over all questions and calculate the metric scores
for question in tqdm(questions):
    question_embedding = question["embedding"]
    similarities = []

    for doc in doc_collection:
        doc_embedding = doc["embedding"]
        similarity = cosine_similarity([question_embedding], [doc_embedding])[0][0]
        similarities.append(similarity)

    # Sort the doc_collection objects based on cosine similarity
    similarity_docs = zip(similarities, doc_collection)
    top_10 = sorted(similarity_docs, key=lambda x: x[0], reverse=True)[:10]

    top_10_ids = [t[1]["id"] for t in top_10]
    retrieved_docs.append(top_10_ids)
    relevant_docs.append(question["paragraphs"])

    precisions.append(precision(top_10_ids, question["paragraphs"]))
    recalls.append(recall(top_10_ids, question["paragraphs"]))
    f1s.append(f1_score(top_10_ids, question["paragraphs"]))

mrr = mean_reciprocal_rank(retrieved_docs, relevant_docs)
map = average_precision(retrieved_docs, relevant_docs)

print("Precision: " + str(sum(precisions) / len(precisions)))
print("Recall: " + str(sum(recalls) / len(recalls)))
print("F1 Score: " + str(sum(f1s) / len(f1s)))
print("MRR: " + str(mrr))
print("MAP: " + str(map))
