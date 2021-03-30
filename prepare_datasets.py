import json
from collections import Counter
import re
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity
from easse.sari import corpus_sari

from utils import get_word_tokens, mean_pooling, get_similarities, get_rougel


if __name__ == "__main__":
    DATA_DIR = Path("./data")
    WIKI_DIR = DATA_DIR / "WikiSimple-translated"

    DEV_PATH = DATA_DIR / "dev_sents.csv"
    TEST_PATH = DATA_DIR / "public_test_only.csv"

    OUTPUT_DIR = Path(DATA_DIR / "prepared_data")
    OUTPUT_DIR.mkdir(exist_ok=True)

    MAX_LENGTH = 200

    # model for embeddings
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model.to("cuda")

    # validation dataset
    dev_df = pd.read_csv(DEV_PATH, index_col=0)
    dev_df.columns = ["input", "output"]

    dev_df["cosine_sim"] = dev_df.apply(lambda x: get_similarities(model, tokenizer, x["input"], x["output"]),
                                    axis=1)
    dev_df["cosine_sim"] = dev_df["cosine_sim"].apply(lambda x: x[0][0])
    dev_df["rouge_l"] = dev_df.apply(lambda x: get_rougel(x["input"], x["output"]), axis=1)
    dev_df["input_len"] = dev_df["input"].apply(lambda x: len(get_word_tokens(x)))
    dev_df["output_len"] = dev_df["output"].apply(lambda x: len(get_word_tokens(x)))
    dev_df.to_csv(OUTPUT_DIR / "dev_df_metrics.csv", index=False)

    # train dataset
    dfs = [pd.read_csv(path, usecols=["target_x", "target_y"]) for path in WIKI_DIR.glob("*")]
    wiki_df = pd.concat(dfs).reset_index(drop=True)
    wiki_df.columns = ["input", "output"]

    wiki_df["cosine_sim"] = wiki_df.apply(lambda x: get_similarities(model, tokenizer, x["input"], x["output"]),
                                        axis=1)
    wiki_df["cosine_sim"] = wiki_df["cosine_sim"].apply(lambda x: x[0][0])

    wiki_df["rouge_l"] = wiki_df.apply(lambda x: get_rougel(x["input"], x["output"]), axis=1)

    wiki_df["input_len"] = wiki_df["input"].apply(lambda x: len(get_word_tokens(x)))
    wiki_df["output_len"] = wiki_df["output"].apply(lambda x: len(get_word_tokens(x)))

    # select data
    wiki_df_part = wiki_df[
        (wiki_df["cosine_sim"] < 0.99)
    & (wiki_df["cosine_sim"] > 0.6)
    & (wiki_df["rouge_l"] < 0.8)
    & (wiki_df["rouge_l"] > 0.1)
    & (wiki_df["output_len"] <= wiki_df["input_len"])
    ]

    wiki_df_part[["input", "output"]].to_csv(OUTPUT_DIR / "wiki_part.csv", index=False)
