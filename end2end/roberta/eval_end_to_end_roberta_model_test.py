# -*- coding: utf-8 -*-


from tqdm import tqdm

from transformers import BertTokenizer
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from torch.optim import Adam
from torch import nn
from transformers import RobertaModel
from torch import cat

from transformers import RobertaTokenizer, set_seed

class EndToEndClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.linear_1 = nn.Linear(1536, 1)

        # Freeze bert layers
        if freeze_bert:
            for p in self.roberta.parameters():
                p.requires_grad = False

    def forward(self, description_input, abstract_input):
        _, pooled_output_1 = self.roberta(**description_input, return_dict=False)
        _, pooled_output_2 = self.roberta(**abstract_input, return_dict=False)

        pooled_output = cat((pooled_output_1, pooled_output_2), dim=1)
        return self.linear_1(pooled_output)


class BertDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]
        description = item["ChallengeDescription"]
        abstrast = item["abstract"]

        description_inputs = self.tokenizer(
            description,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
        )

        abstract_inputs = self.tokenizer(
            abstrast,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "description_input_ids": torch.tensor(
                description_inputs["input_ids"], dtype=torch.long
            ),
            "description_attention_mask": torch.tensor(
                description_inputs["attention_mask"], dtype=torch.long
            ),

            "abstract_input_ids": torch.tensor(
                abstract_inputs["input_ids"], dtype=torch.long
            ),
            "abstract_attention_mask": torch.tensor(
                abstract_inputs["attention_mask"], dtype=torch.long
            ),
        }

def load_data():
    return pd.read_csv("./data/test_df.csv")


def make_pred_for_eval(model, dataloader):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()

    model.eval()
    pred = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            description_inputs = {
                "input_ids": data["description_input_ids"].to(device),
                "attention_mask": data["description_attention_mask"].to(device),
            }

            abstract_inputs = {
                "input_ids": data["abstract_input_ids"].to(device),
                "attention_mask": data["abstract_attention_mask"].to(device),
            }

            output = model(description_inputs, abstract_inputs)
            output = torch.sigmoid(output)
            pred.append(output.cpu().detach().numpy())
    return pred


def main():
    seed = 78
    test_df = load_data()
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    test_ds = BertDataset(test_df, tokenizer)
    eval_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=8)

    model_name = "roberta-e2"
    print(f"evaluating: {model_name}")
    model = EndToEndClassifier()
    model.load_state_dict(torch.load(f"./models/{seed}/{model_name}"))

    preds = make_pred_for_eval(model, eval_dataloader)
    flattened_preds = np.concatenate(preds).ravel()
    results = pd.concat(
        [test_df, pd.DataFrame(flattened_preds, columns=["pred"])], axis=1
    )

    results.to_csv(f"./results-test/{seed}/{model_name}.csv")
    print(classification_report(test_df["label"].tolist(), flattened_preds > 0.5, digits=4))

main()