# -*- coding: utf-8 -*-

from tqdm import tqdm

from transformers.models.deberta.modeling_deberta import ContextPooler

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

import pandas as pd
import numpy as np

import sklearn.metrics as metrics
from sklearn.metrics import classification_report

from torch import nn
from transformers import DebertaV2Tokenizer, DebertaV2Model, AutoConfig
from torch import cat
from sklearn.metrics.pairwise import cosine_similarity

tqdm.pandas()


def load_data():
    dev_papers = pd.read_csv(
        "./data/emb/dev_papers_df.csv"
    )
    dev_cases_data = pd.read_csv(
        "./data/emb/dev_cases_df.csv"
    )
    return dev_papers, dev_cases_data


class BertDataset(Dataset):
    def __init__(self, df, tokenizer, is_paper=True, max_length=512):
        super().__init__()
        self.df = df
        self.is_paper = is_paper
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]
        text = item["abstract"] if self.is_paper else item["ChallengeDescription"]

        inputs = self.tokenizer(
            text,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
        )

        return {
            "input_ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(inputs["token_type_ids"], dtype=torch.long),
        }


class EndToEndClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super().__init__()
        model_name = 'microsoft/deberta-v3-base'
        conf = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.deberta = DebertaV2Model.from_pretrained(model_name, config=conf)
        self.pooler = ContextPooler(conf)
        self.linear_1 = nn.Linear(1536, 1)

        # Freeze bert layers
        if freeze_bert:
            for p in self.deberta.parameters():
                p.requires_grad = False

    def forward(self, description_input, abstract_input):
        output_1 = self.deberta(**description_input)[0]
        output_2 = self.deberta(**abstract_input)[0]
        pooled_output_1 = self.pooler(output_1)
        pooled_output_2 = self.pooler(output_2)
        
        pooled_output = cat((pooled_output_1, pooled_output_2), dim=1)
        return self.linear_1(pooled_output)

    def get_embeddings(self, input):
        output = self.deberta(**input)[0]
        return self.pooler(output)


def create_embeddings(model, dataloader):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()

    model.eval()
    pred = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            inputs = {
                "input_ids": data["input_ids"].to(device),
                "attention_mask": data["attention_mask"].to(device),
                "token_type_ids": data["token_type_ids"].to(device),
            }

            output = model.get_embeddings(inputs)
            pred.append(output.cpu().detach().numpy())
    return np.concatenate(pred)


def convert_y_to_label(y):
    labels = []
    for item in y:
        if item:
            labels.append("associated_paper")
        else:
            labels.append("irrelevant_paper")
    return labels


def get_optimal_threshold(precision, recall, thresholds):
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = np.argmax(fscore)
    return thresholds[ix]


def eval(df):
    df["cos_sim"] = df.progress_apply(
        lambda x: cosine_similarity([x["paper_embedding"]], [x["case_embedding"]])[0],
        axis=1,
    )
    precision, recall, thresholds = metrics.precision_recall_curve(
        df["type"] == "associated_paper", df["cos_sim"]
    )
    optimal_threshold = get_optimal_threshold(precision, recall, thresholds)
    print("Threshold value is:", optimal_threshold)
    df["pred_is_associated"] = df["cos_sim"].apply(lambda x: x >= optimal_threshold)
    y_test_target = convert_y_to_label(df["type"] == "associated_paper")
    y_test_pred = convert_y_to_label(df["pred_is_associated"])
    print(classification_report(y_test_target, y_test_pred, digits=4))
    return df


def main():
    seed = 43
    dev_papers, dev_cases_data = load_data()

    tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')
    papers_ds = BertDataset(dev_papers, tokenizer)
    papers_dataloader = torch.utils.data.DataLoader(papers_ds, batch_size=8)
    cases_ds = BertDataset(dev_cases_data, tokenizer, is_paper=False)
    cases_dataloader = torch.utils.data.DataLoader(cases_ds, batch_size=8)

    model = EndToEndClassifier()
    model.load_state_dict(
        torch.load(f"./models/{seed}/deberta-v3-e1-e2")
    )

    cases_embeddings = create_embeddings(model, cases_dataloader)
    case_embeddings_df = pd.DataFrame({"case_embedding": cases_embeddings.tolist()})
    dev_cases_data_with_emb = pd.concat(
        [dev_cases_data.reset_index(), case_embeddings_df.reset_index()], axis=1
    )

    papers_embeddings = create_embeddings(model, papers_dataloader)
    papers_embeddings_df = pd.DataFrame({"paper_embedding": papers_embeddings.tolist()})

    dev_papers_with_emb = pd.concat(
        [dev_papers.reset_index(), papers_embeddings_df.reset_index()], axis=1
    )

    eval_df = pd.merge(
        dev_papers_with_emb[["case_id", "type", "abstract", "paper_id", "paper_embedding"]],
        dev_cases_data_with_emb[["ChallengeDescription", "CaseID", "case_embedding"]],
        left_on="case_id",
        right_on="CaseID",
    )

    e = eval(eval_df)
    e[["case_id", "paper_id", "cos_sim", "pred_is_associated"]].to_csv(
        f"./results/{seed}/end_to_end_deberta_embeddings_eval.csv"
    )

main()