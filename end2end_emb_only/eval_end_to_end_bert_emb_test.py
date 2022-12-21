# -*- coding: utf-8 -*-

from tqdm import tqdm

from transformers import BertTokenizer

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

import pandas as pd
import numpy as np

import sklearn.metrics as metrics
from sklearn.metrics import classification_report

from torch import nn
from transformers import BertModel
from torch import cat
from sklearn.metrics.pairwise import cosine_similarity

tqdm.pandas()


def load_data():
    test_papers = pd.read_csv(
        "./data/emb/test_papers_df.csv"
    )
    test_cases_data = pd.read_csv(
        "./data/emb/test_cases_df.csv"
    )
    return test_papers, test_cases_data


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
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.linear_1 = nn.Linear(1536, 1)

        # Freeze bert layers
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, description_input, abstract_input):
        _, pooled_output_1 = self.bert(**description_input, return_dict=False)
        _, pooled_output_2 = self.bert(**abstract_input, return_dict=False)

        pooled_output = cat((pooled_output_1, pooled_output_2), dim=1)
        return self.linear_1(pooled_output)

    def get_embeddings(self, input):
        _, pooled_output_1 = self.bert(**input, return_dict=False)
        return pooled_output_1


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

def eval(df, optimal_threshold):
    df["cos_sim"] = df.progress_apply(
        lambda x: cosine_similarity([x["paper_embedding"]], [x["case_embedding"]])[0],
        axis=1,
    )
    df["pred_is_associated"] = df["cos_sim"].apply(lambda x: x >= optimal_threshold)
    y_test_target = convert_y_to_label(df["type"] == "associated_paper")
    y_test_pred = convert_y_to_label(df["pred_is_associated"])
    print(classification_report(y_test_target, y_test_pred, digits=4))
    return df


def main():
    seed = 1337
    threshold = 0.00410117
    test_papers, test_cases_data = load_data()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    papers_ds = BertDataset(test_papers, tokenizer)
    papers_dataloader = torch.utils.data.DataLoader(papers_ds, batch_size=8)
    cases_ds = BertDataset(test_cases_data, tokenizer, is_paper=False)
    cases_dataloader = torch.utils.data.DataLoader(cases_ds, batch_size=8)

    model = EndToEndClassifier()
    model.load_state_dict(
        torch.load(f"./models/{seed}/e2")
    )

    cases_embeddings = create_embeddings(model, cases_dataloader)
    case_embeddings_df = pd.DataFrame({"case_embedding": cases_embeddings.tolist()})
    test_cases_data_with_emb = pd.concat(
        [test_cases_data.reset_index(), case_embeddings_df.reset_index()], axis=1
    )

    papers_embeddings = create_embeddings(model, papers_dataloader)
    papers_embeddings_df = pd.DataFrame({"paper_embedding": papers_embeddings.tolist()})

    test_papers_with_emb = pd.concat(
        [test_papers.reset_index(), papers_embeddings_df.reset_index()], axis=1
    )
    eval_df = pd.merge(
        test_papers_with_emb[["case_id", "type", "abstract", "paper_id", "paper_embedding"]],
        test_cases_data_with_emb[["ChallengeDescription", "CaseID", "case_embedding"]],
        left_on="case_id",
        right_on="CaseID",
    )

    e = eval(eval_df, threshold)
    e[["case_id", "paper_id", "cos_sim", "pred_is_associated"]].to_csv(
        f"./results-test/{seed}/end_to_end_embeddings_eval.csv"
    )


main()