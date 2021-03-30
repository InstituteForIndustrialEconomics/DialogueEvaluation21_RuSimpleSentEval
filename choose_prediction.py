import json
import argparse
from collections import Counter
import re
from itertools import zip_longest
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity
from easse.sari import corpus_sari
from sklearn.ensemble import RandomForestRegressor

from utils import get_word_tokens, mean_pooling, get_similarities, get_rougel


DATA_DIR = Path("./data")
WIKI_DIR = DATA_DIR / "WikiSimple-translated"
DEV_PATH = DATA_DIR / "dev_sents.csv"
TEST_PATH = DATA_DIR / "public_test_only.csv"


def get_preds_df(preds, input_texts):
    preds_df = pd.DataFrame({"pred": preds.values.reshape(-1, 1).ravel()})
    preds_df["input"] = np.array([[i] * 5 for i in input_texts]).ravel()
    preds_df["pred_len"] = preds_df["pred"].apply(lambda x: len(get_word_tokens(x)))
    preds_df["input_len"] = preds_df["input"].apply(lambda x: len(get_word_tokens(x)))
    preds_df["cosine_sim"] = preds_df.apply(lambda x: get_similarities(model, tokenizer, x["pred"], x["input"]),
                                        axis=1)
    preds_df["cosine_sim"] = preds_df["cosine_sim"].apply(lambda x: x[0][0])
    preds_df["rouge_l"] = preds_df.apply(lambda x: get_rougel(x["pred"], x["input"]), axis=1)
    return preds_df


def get_rf_from_dev(dev_df, preds_dev, max_depth=None, random_state=19):
    preds_df = preds_dev.copy()
    dev_df_grouped = dev_df.groupby("input").agg(
        {"output": list, "cosine_sim": list, "rouge_l": list, "input_len": max, "output_len": list}
    ).reset_index()
    preds_df["ref"] = [l for sublist in dev_df_grouped["output"].apply(lambda x: [x] * 5).tolist() for l in sublist]
    preds_df["ref"] = preds_df["ref"].apply(lambda x: [[i] for i in x])

    preds_df["pred_len"] = preds_df["pred"].apply(lambda x: len(get_word_tokens(x)))
    preds_df["input_len"] = preds_df["input"].apply(lambda x: len(get_word_tokens(x)))

    preds_df["sari"] = preds_df.apply(
        lambda x: corpus_sari(
            orig_sents=[x["input"]],
            sys_sents=[x["pred"]],
            refs_sents=x["ref"],
        ), axis=1
    )
    
    rf = RandomForestRegressor(n_estimators=1000, max_depth=max_depth, n_jobs=-1, random_state=random_state)
    
    X_train = preds_df[["cosine_sim", "rouge_l", "input_len", "pred_len"]]
    y_train = preds_df["sari"]
    
    rf.fit(X_train, y_train)
    
    return rf, preds_df


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_test_path", required=True, type=str, default="data/public_test_only.csv")
    argparser.add_argument("--input_dev_metrics_path", required=True, type=str,
                           default="data/prepared_data/dev_df_metrics.csv")
    argparser.add_argument("--test_predictions_path", required=True, type=str, default="predictions/test_answers_5.csv")
    argparser.add_argument("--dev_predictions_path", required=True, type=str, default="predictions/dev_answers_5.csv")
    argparser.add_argument("--submission_folder", required=True, type=str, default="submissions")
    argparser.add_argument("--seed", required=True, type=int, default=19)
    args = argparser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model.to("cuda")

    with open(args.input_test_path) as f:
        test_input = [l.strip() for l in f]

    dev_df = pd.read_csv(args.input_dev_metrics_path)
    dev_df_grouped = dev_df.groupby("input").agg(
        {"output": list, "cosine_sim": list, "rouge_l": list, "input_len": max, "output_len": list}
    ).reset_index()
    dev_input = dev_df_grouped["input"].tolist()

    pred_test = pd.read_csv(args.test_predictions_path)
    pred_test = get_preds_df(pred_test, test_input)

    preds_dev = pd.read_csv(args.dev_predictions_path)
    preds_dev = get_preds_df(preds_dev, dev_input)

    rf, preds_df = get_rf_from_dev(dev_df, preds_dev, max_depth=5, random_state=args.seed)

    pred_test["sari_pred"] = rf.predict(pred_test[["cosine_sim", "rouge_l", "input_len", "pred_len"]])
    tmp = pred_test.sort_values(["input", "sari_pred"], ascending=[True, False])[::5]
    pred_dict = dict(zip(tmp["input"], tmp["pred"]))

    predictions = [pred_dict[i] for i in test_input]

    submission_folder = Path(args.submission_folder)
    submission_folder.mkdir(exist_ok=True)
    pd.DataFrame(predictions).to_csv(submission_folder / "answer.csv", index=False, header=None)
