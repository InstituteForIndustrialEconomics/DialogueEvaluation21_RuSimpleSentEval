import argparse
import json
import random
from pprint import pprint
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          Trainer,
                          TrainingArguments,
                          TrainerCallback,
                          AdamW,
                          get_linear_schedule_with_warmup)

from utils import set_random_seed


def get_shortest_prediction(text, args):
    prompt = f"<|startoftext|>{text}<|sep|>"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(args.device)
    sample_outputs = model.generate(
        input_ids,
        do_sample=True,   
        top_k=50, 
        max_length = MAX_LENGTH,
        top_p=0.95,
        temperature=0.9,
        length_penalty=0.7,
        num_return_sequences=5,
    ).detach().cpu()

    predictions = []
    for sample in sample_outputs:
        result = (tokenizer.decode(sample, skip_special_tokens=False)
                        .split("<|sep|>")[1]
                        .replace("<|pad|>", "")
                        .replace("<|endoftext|>", ""))
        predictions.append(result)
    prediction = sorted(predictions, key=lambda x: len(x))[0]

    return prediction


def get_predictions(text, args):
    prompt = f"<|startoftext|>{text}<|sep|>"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(args.device)
    sample_outputs = model.generate(
        input_ids,
        do_sample=True,   
        top_k=50, 
        max_length = MAX_LENGTH,
        top_p=0.95,
        temperature=0.9,
        length_penalty=0.7,
        num_return_sequences=5,
    ).detach().cpu()

    predictions = []
    for sample in sample_outputs:
        result = (tokenizer.decode(sample, skip_special_tokens=False)
                        .split("<|sep|>")[1]
                        .replace("<|pad|>", "")
                        .replace("<|endoftext|>", ""))
        predictions.append(result)

    return predictions


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_test_path", required=True, type=str, default="data/public_test_only.csv")
    argparser.add_argument("--input_dev_path", required=True, type=str, default="data/dev_sents.csv")
    argparser.add_argument("--model_path", required=True, type=str, default="result_rugpt3medium/checkpoint-45282")
    argparser.add_argument("--predict_folder", required=True, type=str, default="predictions")
    argparser.add_argument("--seed", required=True, type=int, default=19)
    argparser.add_argument("--device", required=True, type=str, default="cuda:0")
    args = argparser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    MAX_LENGTH = 200
    set_random_seed(args.seed)
    predict_folder = Path(args.predict_folder)
    predict_folder.mkdir(parents=True, exist_ok=True)

    # load model
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.to(args.device)
    # cannot use AutoTokenizer due to some bug
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")
    special_tokens = {
        "bos_token": "<|startoftext|>",
        "pad_token": "<|pad|>",
        "sep_token": "<|sep|>",
    }
    tokenizer.add_special_tokens(special_tokens)

    # load test data
    with open(args.input_test_path) as f:
        test = [line.strip() for line in f]

    predictions = []
    for text in tqdm(test):
        pred = get_predictions(text, args)
        predictions.append(pred)

    pd.DataFrame({"pred": predictions})["pred"].apply(pd.Series).to_csv(predict_folder / "test_answers_5.csv",
                                                                        index=False)

    # load dev data
    dev = pd.read_csv(args.input_dev_path)
    dev = dev.drop_duplicates("INPUT:source")

    predictions = []
    for text in tqdm(dev["INPUT:source"].values):
        pred = get_predictions(text, args)
        predictions.append(pred)

    pd.DataFrame({"pred": predictions})["pred"].apply(pd.Series).to_csv(predict_folder / "dev_answers_5.csv",
                                                                        index=False)
