# This file quries openai's GPT-3.5-turbo model to generate answers based on the layperson's questions
# together with the respective relevant paragraphs for the layquestion

import json
import pandas as pd
import openai
import tiktoken
from rouge import Rouge 
from nltk.translate.bleu_score import sentence_bleu
import bert_score
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# ### 1. Load questions with its gold_standards and gold_answers
# - Usage of the 50 best results from the retrieval process => reduce amount of examples due to billing
# - Translate them to DataPoint objects

class DataPoint:
    def __init__(self, qt:str, gold_paras: [str], gold_answer: str, model_answer: str = "", bleu: float = 0.0, rouge: float = 0.0, bert_score: float = 0.0, sem_sim: float = 0.0, para_gold_answ = ""):
        self.question_text = qt
        self.gold_paragraphs = gold_paras
        self.gold_answer = gold_answer
        self.generated_answer = model_answer
        self.bleu = bleu
        self.rouge = rouge
        self.bert_score = bert_score
        self.semantic_similarity = sem_sim
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
        "rouge": self.rouge,
        "bert_score": self.bert_score,
        }

    @classmethod
    def from_dict(cls, dict_data):
        return cls(dict_data["question_text"], dict_data["gold_paragraphs"], dict_data["gold_answer"], dict_data["generated_answer"], dict_data["rouge"], dict_data["bert_score"], dict_data["paraphrased_gold_answer"])
    
    def to_query(self):
        return f"Beantworte folgende Frage: {self.question_text} \nNutze zur Beantwortung folgende Gesetztestexte: \n" + "\n".join(self.gold_paragraphs)

    def to_paraphrase_query(self):
        return f"Schreibe diesen Text um, aber behalte alle Informationen! \n {self.gold_answer}"


# list to keep our data points
datapoints: [DataPoint] = []

# only run, if no ai generated answer was stored in the file yet.
# If openai api was already accessed, the answers are stored in the file and can therefore be accessed by simply loading the file which is done under Section 4
with open("./data/bgb_eval", "r") as f:
    data = json.load(f)

for x in data:
          datapoints.append(DataPoint(x[0], x[2], x[1]))


# 2. Access OpenAI's GPT3.5-turbo API
# - for each of the data pairs access the api with the following format: "Beantworte diesen Sachverhalt : {question_text} vor folgendem Hintergrund: {gold1_text} \n {gold2_text}..."
# - save the openAi's generated answer within the DataPoint objects and create a json file with the results
# add your api key
openai.api_key = ""

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def send_to_openai_api_4k(prompt: str, max_tokens) -> str:
    'sends a prompt to the openai 4k context model and returns its answer'
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [{"role": "user", "content": prompt}],
        max_tokens = max_tokens
    )

    return response.choices[0].message["content"]

def send_to_openai_api_16k(prompt: str, max_tokens) -> str:
    'sends a prompt to the openai 16k context model and returns its answer'
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo-16k",
        messages = [{"role": "user", "content": prompt}],
        max_tokens = max_tokens
    )

    return response.choices[0].message["content"]


def saveDatapoints():
    'save the datapoints in bgb_eval_qa.json'
    with open("./bgb_eval_qa.json", "w") as f:
        datapoints_dict_list = [x.data_point_to_dict() for x in datapoints]
        json.dump(datapoints_dict_list, f)

# define max tokens for answer
max_answer_tokens = 1500

# some counters to see how often the big model was used
counter = 0
bigmodel_counter = 0
smallmodel_counter = 0

for dp in tqdm(datapoints):
    if dp.generated_answer == "":
        try:
            if (num_tokens_from_string(dp.to_query()) + max_answer_tokens > 4096):
                res = send_to_openai_api_16k(dp.to_query(), max_answer_tokens)
                dp.generated_answer= res.replace("\n", "")
                bigmodel_counter += 1
            else:
                res = send_to_openai_api_4k(dp.to_query(), max_answer_tokens)
                dp.generated_answer= res.replace("\n", "")
                smallmodel_counter += 1
                print(f"Error with: {counter}")
        except:
            # try a second time with the bigger model, seemed to work on second try sometimes
            res = send_to_openai_api_16k(dp.to_query(), max_answer_tokens)
            dp.generated_answer= res.replace("\n", "")
            bigmodel_counter += 1


    # get process and also store intermediate state for potential crashes
    if counter % 25 == 0:
        saveDatapoints()
        print(f"big model used {bigmodel_counter}/{len(datapoints)}")
    
    counter += 1

# finally save the datapoints
saveDatapoints()