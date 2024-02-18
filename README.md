# Answering legal questions from laymen in German civil law system: Data and code
This repository is part of the 2024 published EACL Paper "Answering legal questions from laymen in German civil law system" by Marius Büttner and Prof Dr. Ivan Habernal.

## Data Usage and Copyright Notice
This repository contains data that has been collected exclusively from publicly accessible websites. The process of data collection was conducted with respect to the websites' terms of use and is intended for educational and scientific research purposes only.

Disclaimer: We do not claim ownership or hold any copyrights over the data provided herein. All data remains the property of their respective owners. The data shared in this repository is offered without any warranties, and users are solely responsible for its use.

Usage Restrictions: The data hosted in this repository is strictly for non-commercial, scientific research purposes. Any use of this data for commercial purposes is expressly prohibited.

## Repository setup:

### Python files:
- multi_thread_data_scraper.py
    - Part of the papers Data scraping section (File 1/1)
    - This file was used to scrape the raw data for our dataset 

- doc_retrieval_sentence_transformers_baselines.py
    - Part of paper's Document Retrieval Section (File 1/2)
    - This file is used to calculate the F1, Precision, Recall, MRR and MAP scores of a Hugging Face sentence transformers model against the bgb_eval dataset for the task of document retrieval

- doc_retrieval_open_ai_baseline
    - Part of paper's Document Retrieval Section (File 2/2)
    - This file is used to calculate the F1, Precision, Recall, MRR and MAP scores of the openAI GPT-3.5-turbo model against the bgb_eval dataset for the task of document retrieval

- answer_generation_prep.py:
    - Part of the paper's Answer Generation Section (File 1/2)
    - This file queries OpenAI's GPT-3.5-turbo model to generate answers based on the layperson's questions and relevant paragraphs

- answer_generation_eval.py:
    - Part of paper's Answer Generation Section (File 2/2) 
    - We calculate the upper, lower borders and the actual model performance for our ROUGE, BERTScore.


### Data and datasets:
Files in the data subfolder:
- GerLayQA.json
    - Primary dataset with questions and answers targeting at the German BGB
- bgb_dev.json
    - Our splitted development part of the GerLayQA dataset
- bgb_eval.json
    - Our splitted evaluation part of the GerLayQA dataset
- bgb_train.json
    - Our splitted training part of the GerLayQA dataset
- bgb.json
    - Contains all (non-changed) paragraphs of the German BGB, is used as document collection within the document retrieval task
- stgb_QA.json
    - Dataset with questions and answers targeting at the German StGB
- zpo_QA.json
    - Dataset with questions and answers targeting at the German ZPO




