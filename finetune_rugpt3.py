import argparse
import json
import random
from pprint import pprint
from pathlib import Path
import pandas as pd
import numpy as np
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


def get_random_example(dataset):
    """
    Get random example from given dataset.
    Return promt with special tokens and true output.
    """
    sample = dataset.sample()
    prompt = f'<|startoftext|>{sample["input"].item()}<|sep|>'
    true_output = sample["output"].item()
    return prompt, true_output


class SimplificationDataset(torch.utils.data.Dataset):
    def __init__(self, texts_list, tokenizer, gpt2_type="gpt2", max_length=1024):
        self.tokenizer = tokenizer

        texts_combined = []
        for input_text, out_text in texts_list:
            text_combined = f"<|startoftext|>{input_text}<|sep|>{out_text}<|endoftext|>"
            texts_combined.append(text_combined)
        self.encodings = tokenizer(texts_combined,
                              truncation=True,
                              max_length=max_length,
                              padding="max_length",
                              return_tensors="pt")

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = item["input_ids"]
        return item


class PrintExampleCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        prompt, true_output = get_random_example(valid)
        print(prompt.strip("<|startoftext|>").strip("<|sep|>"), true_output, sep="\n")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        model.eval()
        with torch.no_grad():
            sample_outputs = model.generate(
                input_ids,
                do_sample=True,   
                top_k=50,
                max_length=MAX_LENGTH,
                top_p=0.95,
                temperature=0.9,
                num_return_sequences=1
            ).detach().cpu()
        model.train()

        for sample in sample_outputs:
            res = (tokenizer.decode(sample, skip_special_tokens=False)
                            .split("<|sep|>")[1]
                            .replace("<|pad|>", "")
                            .replace("<|endoftext|>", ""))
            print(res, "-" * 80, sep="\n")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", required=True, type=str, default="wiki_part.csv")
    argparser.add_argument("--model", required=True, type=str, default="sberbank-ai/rugpt3medium_based_on_gpt2")
    argparser.add_argument("--output_dir", required=True, type=str, default="result_rugpt3medium")
    argparser.add_argument("--log_dir", required=True, type=str, default="logs_rugpt3medium")
    argparser.add_argument("--epochs", required=True, type=int, default=5)
    argparser.add_argument("--batch_size", required=True, type=int, default=8)
    argparser.add_argument("--gradient_accumulation_steps", required=True, type=int, default=1)
    argparser.add_argument("--learning_rate", required=True, type=float, default=0.00005)
    argparser.add_argument("--warmup_steps", required=True, type=int, default=500)
    argparser.add_argument("--seed", required=True, type=int, default=19)
    argparser.add_argument("--fp16", required=False, dest="fp16", default=False, action="store_true")
    args = argparser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    set_random_seed(args.seed)

    # load train and valid data
    DATA_DIR = Path("./data")
    TRAIN_PATH = DATA_DIR / "prepared_data" / args.dataset
    VALID_PATH = DATA_DIR / "dev_sents.csv"
    TEST_PATH = DATA_DIR / "public_test_only.csv"

    train = pd.read_csv(TRAIN_PATH)
    valid = pd.read_csv(VALID_PATH, index_col=0)
    valid.columns = ["input", "output"]

    # load model
    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # add special tokens
    special_tokens = {
        "bos_token": "<|startoftext|>",
        "pad_token": "<|pad|>",
        "sep_token": "<|sep|>",
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    MAX_LENGTH = 200  # from EDA
    DATA_COLS = ["input", "output"]
    train_dataset = SimplificationDataset(train[DATA_COLS].values.tolist(), tokenizer, max_length=MAX_LENGTH)
    valid_dataset = SimplificationDataset(valid[DATA_COLS].values.tolist(), tokenizer, max_length=MAX_LENGTH)

    EPOCH_STEPS = len(train_dataset) // args.batch_size // args.gradient_accumulation_steps
    EVAL_STEPS = EPOCH_STEPS // 2  # save 2 times per epoch
    print(f"Total steps: {EPOCH_STEPS * args.epochs}\nEvaluate and save every {EVAL_STEPS} steps.")

    Path(args.output_dir).mkdir(exist_ok=True)
    Path(args.log_dir).mkdir(exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.log_dir,
        logging_first_step=True,
        num_train_epochs=args.epochs,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=EVAL_STEPS,
        logging_steps=100,
        lr_scheduler_type="linear",
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=0,
        fp16=args.fp16,
        seed=args.seed,
    )

    pprint(training_args.to_dict())
    with open(Path(args.output_dir) / "run_parameters.txt", "w") as f:
        pprint(training_args.to_dict(), f)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[PrintExampleCallback],
    )

    trainer.train()
