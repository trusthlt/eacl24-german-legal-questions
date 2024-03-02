# Here, we calculate the upper, lower borders and the actual model performance for our ROUGE, BERTScore.
# 
# ### Lower border:
# - We want to compare the original legal paragraph that is relevant to the answer rephrased by the lawyer (gold_answer)
# 
# ### Upper border:
# - We paraphrase the lawyers' answers for all examples by telling the openAI api to do so
# - Query: "Schreibe diesen Text um, aber behalte alle Informationen \n {gold_answer}
# - We store that query in our DataPoint class
# - We calculate the scores for all of our metrics by using the gold_answer and the rephrased_gold_answer, and then calculate the average of them

import json
import pandas as pd
import random
from rouge import Rouge 
from sentence_transformers import SentenceTransformer, util
from evaluate import load
bertscore = load("bertscore")


# ### 1 Setup and Load DataPoints from file
class DataPoint:
    'Represents a datapoint from our dataset'
    def __init__(self, qt:str, gold_paras: [str], gold_answer: str, model_answer: str = "", bleu: float = 0.0, rouge: float = 0.0, bert_score: float = 0.0, sem_sim: float = 0.0, para_gold_answ = ""):
        self.question_text = qt
        self.gold_paragraphs = gold_paras
        self.gold_answer = gold_answer
        self.generated_answer = model_answer
        self.rouge = rouge
        self.bert_score = bert_score
        self.paraphrased_gold_answer = para_gold_answ

    def toString(self):
        print("Question_text      : ", self.question_text)
        print("Gold_Paragraphs    : ", self.gold_paragraphs)
        print("Gold_answer        : ", self.gold_answer)
        print("Paraphrased Gold Answ.: ", self.paraphrased_gold_answer)
        print("Generated_Answer   : ", self.generated_answer)
        print("ROUGE_Score        : ", self.rouge)
        print("BERTScore          : ", self.bert_score)
            
    def data_point_to_dict(self):
        return {
        "question_text": self.question_text,
        "gold_paragraphs": self.gold_paragraphs,
        "gold_answer": self.gold_answer,  
        "paraphrased_gold_answer" : self.paraphrased_gold_answer,
        "generated_answer": self.generated_answer,
        "bleu": self.bleu,
        "rouge": self.rouge,
        "bert_score": self.bert_score,
        "semantic_similarity": self.semantic_similarity
        }

    @classmethod
    def from_dict(cls, dict_data):
        return cls(dict_data["question_text"], dict_data["gold_paragraphs"], dict_data["gold_answer"], dict_data["generated_answer"], dict_data["rouge"], dict_data["bert_score"], dict_data["paraphrased_gold_answer"])
    
    def to_query(self):
        return f"Beantworte folgende Frage: {self.question_text} \nNutze zur Beantwortung folgende Gesetztestexte: \n" + "\n".join(self.gold_paragraphs)

    def to_paraphrase_query(self):
        return f"Schreibe diesen Text um, aber behalte alle Informationen! \n {self.gold_answer}"


with open("./data/bgb_eval_qa.json", "r") as f:
    datapoints_dicts = json.load(f)
    datapoints = [DataPoint.from_dict(dp_dict) for dp_dict in datapoints_dicts]


def saveDatapoints():
    with open("./bgb_eval_qa.json", "w") as f:
        datapoints_dict_list = [x.data_point_to_dict() for x in datapoints]
        json.dump(datapoints_dict_list, f)


# 2 Compute Borders scores
# Initialize the semantic_similarity model
model = SentenceTransformer('PM-AI/bi-encoder_msmarco_bert-base_german')

def compute_bleu(reference, candidate):
    """
    Compute the BLEU score between a reference and candidate text.
    """
    reference = [reference.split()]
    candidate = candidate.split()
    return sentence_bleu(reference, candidate)

def compute_rouge(reference, candidate):
    """
    Compute the ROUGE score between a reference and candidate text.
    Returns a dictionary with ROUGE-1, ROUGE-2 and ROUGE-L scores.
    """
    return rouge.compute(predictions=[dp.question_text], references=[dp.gold_answer])


def compute_bertscore(reference, candidate):
    """
    Compute the BERTScore between a reference and candidate text.
    Returns Precision, Recall, and F1 scores.
    """
    results = bertscore.compute(predictions=[candidate], references=[reference], lang="de")

    return (results["precision"][0], results["recall"][0], results["f1"][0])

def compute_similarity(sentence1, sentence2):
    # Obtain embeddings
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)
    
    # Compute cosine similarity
    similarity = util.pytorch_cos_sim(embedding1, embedding2)
    return similarity.item()

def calculate_averages():

    num_datapoints = len(datapoints)
    
    return [
    total_rouge_1 / num_datapoints,
    total_rouge_2 / num_datapoints,
    total_rouge_l / num_datapoints,
    total_bertscore_f1 / num_datapoints,]
    

def print_totals_table():
        
    ## Display the results as table
    metrics = {
        "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore"],
        "Score": calculate_averages()
    }

    table = "| Metric               | Score      |\n"
    table += "|----------------------|------------|\n"
    for metric, score in zip(metrics["Metric"], metrics["Score"]):
        table += f"| {metric:20} | {score:.4f} |\n"

    print(table)


# ### Calculating the Upper boundary
# We calculate the average scores over the metris for comparing the gold_answer to our  paraphrased gold_answer of the datapoint.  
# We can use this than as upper border as we have a text that for sure includes all the information from our gold_answer

total_rouge_1 = 0
total_rouge_2 = 0
total_rouge_l = 0
total_bertscore_f1 = 0

# Compute metrics for each datapoint
for dp in datapoints:
    rouge_scores = compute_rouge(dp.gold_answer, dp.paraphrased_gold_answer)
    total_rouge_1 += rouge_scores['rouge-1']
    total_rouge_2 += rouge_scores['rouge-2']
    total_rouge_l += rouge_scores['rouge-l']

    _, _, bert_score_f1 = compute_bertscore(dp.gold_answer, dp.paraphrased_gold_answer)
    
    total_bertscore_f1 += bert_score_f1

print("Upper border")
print_totals_table()


# ### Lower Boundary
# We will use the law paragraphs (concatinated) and compare them with the gold_answers  
# first: reset our totals

total_rouge_1 = 0
total_rouge_2 = 0
total_rouge_l = 0
total_bertscore_f1 = 0

random.seed(42)

# Compute metrics for each datapoint
for dp in datapoints:
    rouge_scores = compute_rouge(dp.gold_answer, " ".join(dp.gold_paragraphs))
    total_rouge_1 += rouge_scores['rouge-1']
    total_rouge_2 += rouge_scores['rouge-2']
    total_rouge_l += rouge_scores['rouge-l']

    _, _, bert_score_f1 = compute_bertscore(dp.gold_answer, " ".join(dp.gold_paragraphs))
    total_bertscore_f1 += bert_score_f1

print("Lower Border")
print_totals_table()


# ### Actual Performance
# Calculate metrics for the actual Results
# first: reset our totals

total_rouge_1 = 0
total_rouge_2 = 0
total_rouge_l = 0
total_bertscore_f1 = 0

# Compute metrics for each datapoint
for dp in datapoints:
    
    dp.rouge = compute_rouge(dp.gold_answer, dp.generated_answer)
    total_rouge_1 += rouge_scores['rouge-1']
    total_rouge_2 += rouge_scores['rouge-2']
    total_rouge_l += rouge_scores['rouge-l']

    bert_score_p, bert_score_r, bert_score_f1 = compute_bertscore(dp.gold_answer, dp.generated_answer)
    dp.bert_score = {"Precision": bert_score_p, "Recall": bert_score_r, "F1": bert_score_f1}

    total_bertscore_f1 += bert_score_f1  

print("Actual Results")
print_totals_table()

