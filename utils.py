import json
from collections import Counter
import re
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import random
from transformers import AutoModel, AutoTokenizer
from rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity
from easse.sari import corpus_sari


MAX_LENGTH = 200


def get_word_tokens(text):
    tokens = re.sub(r"[^\w\s]", "", text).split()
    tokens = [t.lower() for t in tokens]
    return tokens


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def get_similarities(model, tokenizer, input_texts, output_texts):
    # Tokenize sentences
    encoded_input = tokenizer(input_texts, padding=True, truncation=True,
                              max_length=MAX_LENGTH, return_tensors="pt").to(model.device)
    encoded_output = tokenizer(output_texts, padding=True, truncation=True,
                              max_length=MAX_LENGTH, return_tensors="pt").to(model.device)

    # Compute token embeddings
    with torch.no_grad():
        model_emb_input = model(**encoded_input)
        model_emb_output = model(**encoded_output)

    # Perform pooling. In this case, mean pooling
    input_embeddings = mean_pooling(model_emb_input, encoded_input["attention_mask"]).cpu()
    output_embeddings = mean_pooling(model_emb_output, encoded_output["attention_mask"]).cpu()
    similarity = cosine_similarity(input_embeddings, output_embeddings)
    return similarity


def get_rougel(input_text, output_text):
    """
    Returns rouge-l f-score
    """
    rouge = Rouge()
    scores = []
    # try/except because of empty output or just dot (in dev_sents)
    try:
        score = rouge.get_scores(input_text, output_text)[0]
        score = score["rouge-l"]["f"]
    except ValueError:  
        score = 0.0
    return score


def set_random_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
