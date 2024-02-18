## This file is used to calculate the F1, Precision, Recall, MRR and MAP scores of a Hugging Face sentence transformers model against the bgb_eval dataset for the task of document retrieval

import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util, InputExample, losses, evaluation, models
from torch.utils.data import DataLoader
from tqdm import tqdm

def load_data_collection(sourcefile, max_token_length):
    """loads the doc collection and splits it if necessary"""
    with open(sourcefile, "r") as f:
        data = json.load(f)

    for doc in data:
        length = len(doc["content"].split(" "))
        if length > max_token_length:
            content = doc["content"].split(" ")
            iterations = length // 400
            doc["content"] = " ".join(content[0:400])

            split_strings = [' '.join(content[i:i+400])
                             for i in range(1, length, 400)]
            for s in split_strings:
                data.append({"content": s, "id": doc["id"]})

    documents = [doc["content"] for doc in data]
    doc_content_to_id = {doc["content"]: doc["id"] for doc in data}
    doc_id_to_content = {doc["id"]: doc["content"] for doc in data}

    return (documents, doc_content_to_id, doc_id_to_content)


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
    if (p+r == 0):
        return 0

    return 2 * (p * r) / (p + r)

def mrr_score(retrieved_docs, expected_docs):
    """calculates the Mean Reciprocal Rank (MRR)"""
    reciprocal_ranks = []
    for ret_docs, exp_docs in zip(retrieved_docs, expected_docs):
        for rank, doc in enumerate(ret_docs, start=1):
            if doc in exp_docs:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)

def map_score(retrieved_docs, expected_docs):
    average_precisions = []
    for ret_docs, exp_docs in zip(retrieved_docs, expected_docs):
        if len(exp_docs) == 0:
            # Skip this query if there are no expected documents
            continue

        relevant_docs_found = 0
        precision_sum = 0
        for rank, doc in enumerate(ret_docs, start=1):
            if doc in exp_docs:
                relevant_docs_found += 1
                precision_sum += relevant_docs_found / rank
        if relevant_docs_found > 0:
            average_precisions.append(precision_sum / len(exp_docs))
        else:
            # If no relevant documents were found, add 0 to the average precisions
            average_precisions.append(0)

    # Avoid division by zero by checking if average_precisions is empty
    return sum(average_precisions) / len(average_precisions) if average_precisions else 0


def retrieve_documents(model, question, doc_embeddings, doc_content_to_id, k=10):
    """retrieves the related docs for the questions"""
    question_embedding = model.encode(question, device=device)

    # Compute cosine similarity between question embedding and document embeddings
    cosine_scores = util.pytorch_cos_sim(question_embedding, doc_embeddings)[0]

    # Sort documents based on similarity scores
    sorted_indices = np.argsort(cosine_scores.numpy())[-k:][::-1]

    # Retrieve top-k documents
    top_documents = [documents[i] for i in sorted_indices]

    # transform the documents to its ids
    top_documents = [doc_content_to_id[doc] for doc in top_documents]
    return top_documents


def evaluate_model(model, eval_set, doc_content_to_id, doc_embeddings):
    """
    evaluates the given model in the eval set
    :return: tuple with avg for precision, recall, f1
    """
    precisions, recalls, f1s = [], [], []
    all_retrieved_docs = []
    all_expected_docs = []

    for question in tqdm(eval_set):
        if (len(question["Question_text"].split(" ")) > 300):
            continue

        expected_docs = question["Paragraphs"]
        retrieved_docs = retrieve_documents(
            model, question["Question_text"], doc_embeddings, doc_content_to_id)

        precisions.append(precision(retrieved_docs, expected_docs))
        recalls.append(recall(retrieved_docs, expected_docs))
        f1 = f1_score(retrieved_docs, expected_docs)
        if (f1 != None):
            f1s.append(f1)

        if (len(expected_docs) > 0):
            all_retrieved_docs.append(retrieved_docs)
            all_expected_docs.append(expected_docs)

    mrr = mrr_score(all_retrieved_docs, all_expected_docs)
    map_sc = map_score(all_retrieved_docs, all_expected_docs)

    return (sum(precisions)/len(precisions), sum(recalls)/len(recalls), sum(f1s)/len(f1s), mrr, map_sc)


# call this method to start the metrics calculation
def execute(model, modelName, documents, eval_set, doc_content_to_id, device):
    """starts the calculation of the model's scores"""
    doc_embeddings = model.encode(documents, device=device)

    # Baseline evaluation
    baseline_precision, baseline_recall, baseline_f1_score, baseline_mrr, baseline_map_sc = evaluate_model(
        model, eval_set, doc_content_to_id, doc_embeddings)

    print(f"#### {modelName} performance: ####")
    print("Precision :  " + f"{baseline_precision:.3f}")
    print("Recall    :  " + f"{baseline_recall:.3f}")
    print("F1 Score  :  " + f"{baseline_f1_score:.3f}")
    print("MRR  :  " + f"{baseline_mrr:.3f}")
    print("MAP  :  " + f"{baseline_map_sc:.3f}")


### MAIN ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_token_length = 500

# read the datasets
with open("./data/bgb_dev.json") as f:
    eval_set = json.load(f)

# load the document collection, doc_id_to_content mapping
documents, doc_content_to_id, doc_id_to_content = load_data_collection(
    "./data/bgb.json", max_token_length)

# you can add various models here and calculate the scores for all models you want.
# you need to add them just like the hugging face website says for the sentence_transformer's library
embed_models = [
    ("PM-AI/bi-encoder_msmarco_bert-base_german",
     SentenceTransformer("PM-AI/bi-encoder_msmarco_bert-base_german"))
]

for model in embed_models:
    try:
        print(model)
        execute(model[1], model[0], documents, eval_set, doc_content_to_id, device)
    except Exception as ex:
        print(f"Model validation failed for {model[0]}")
        print(ex)
