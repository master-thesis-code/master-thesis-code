# -*- coding: utf-8 -*-
import pandas as pd
import tqdm
import pathlib

import sklearn.metrics as metrics
import numpy as np
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer, util

tqdm.tqdm.pandas()


def load_data():
    test_df = pd.read_csv(".data/test.csv")
    test_cases_df = pd.read_csv(".data/test_cases.csv")
    return test_df, test_cases_df


def embed(df, input, model):
    embeddings = model.encode(df[input].tolist(), show_progress_bar=True)
    embeddings_df = pd.DataFrame({input + "_embedding": embeddings.tolist()})
    return pd.concat([df, embeddings_df], axis=1)


def convert_y_to_label(y):
    labels = []
    for item in y:
        if item:
            labels.append("associated_paper")
        else:
            labels.append("irrelevant_paper")
    return labels


def eval(df):
    optimal_threshold = 0.00311606
    df["pred_is_associated"] = df["cos_sim"].apply(lambda x: x >= optimal_threshold)
    y_test_target = convert_y_to_label(df["type"] == "associated_paper")
    y_test_pred = convert_y_to_label(df["pred_is_associated"])
    print(classification_report(y_test_target, y_test_pred))
    return df


def eval_model(model, name, seed, dev_df, dev_cases_df):
    dev_emb_df = embed(dev_df, "abstract", model)
    cases_emb_df = embed(dev_cases_df, "ChallengeDescription", model)
    embeded_df = pd.merge(
        dev_emb_df[["case_id", "type", "paper_id", "abstract", "abstract_embedding"]],
        cases_emb_df[
            ["ChallengeDescription", "CaseID", "ChallengeDescription_embedding"]
        ],
        left_on="case_id",
        right_on="CaseID",
    )
    embeded_df["cos_sim"] = embeded_df.progress_apply(
        lambda x: util.cos_sim(
            x["abstract_embedding"], x["ChallengeDescription_embedding"]
        ).item(),
        axis=1,
    )
    eval_df = eval(embeded_df)
    eval_df[["case_id", "paper_id", "cos_sim", "pred_is_associated"]].to_csv(
        f"./results-test/{seed}/{name}.csv"
    )


def main():
    seed = 43
    test_df, test_cases_df = load_data()
    model_path = f"./data/final_model_hard_negatives/{seed}"
    model_name = f"{seed}_final"
    print("Evaluating " + model_name)
    model = SentenceTransformer(model_path)
    eval_model(model, model_name, seed, test_df, test_cases_df)

main()
