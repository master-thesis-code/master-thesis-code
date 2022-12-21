# -*- coding: utf-8 -*-
from tqdm import tqdm

from torch import nn
from torch import cat
from transformers import DebertaV2Tokenizer, DebertaV2Model, AutoConfig
from transformers import set_seed
from transformers.models.deberta.modeling_deberta import ContextPooler
import pandas as pd
import numpy as np

from torch.optim import Adam

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from timeit import default_timer as timer
from datetime import timedelta


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
        label = item["label"]

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
            'description_token_type_ids': torch.tensor(description_inputs["token_type_ids"], dtype=torch.long),

            "abstract_input_ids": torch.tensor(
                abstract_inputs["input_ids"], dtype=torch.long
            ),
            "abstract_attention_mask": torch.tensor(
                abstract_inputs["attention_mask"], dtype=torch.long
            ),
            'abstract_token_type_ids': torch.tensor(abstract_inputs["token_type_ids"], dtype=torch.long),
            "target": torch.tensor(label, dtype=torch.long),
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


def setup_seed(seed):
    torch.manual_seed(seed)
    set_seed(seed)
    np.random.seed(seed)


def train(model, train_dataloader, pos_weight=1, epochs=1, start_epoch=0, seed=404, bert_lr=5e-5, linear_lr=0.001):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam([
        {"params":model.bert.parameters(),"lr": bert_lr},
        {"params":model.linear_1.parameters(), "lr": linear_lr},
   ])

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    total_acc_train = 0
    total_loss_train = 0
    model.train()
    start = timer()
    for epoch in range(0, epochs):
        print(f"Epoch: {epoch}")
        for i, train_data in enumerate(tqdm(train_dataloader)):
            train_label = train_data["target"].unsqueeze(1).to(device)
            description_inputs = {
                "input_ids": train_data["description_input_ids"].to(device),
                "attention_mask": train_data["description_attention_mask"].to(device),
                'token_type_ids': train_data['description_token_type_ids'].to(device)
            }

            abstract_inputs = {
                "input_ids": train_data["abstract_input_ids"].to(device),
                "attention_mask": train_data["abstract_attention_mask"].to(device),
                'token_type_ids': train_data['abstract_token_type_ids'].to(device)
            }

            output = model(description_inputs, abstract_inputs)

            batch_loss = criterion(output.float(), train_label.float())
            loss_value = batch_loss.item()
            total_loss_train += loss_value

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), f"./models/{seed}/deberta-v3-e{epoch + start_epoch}")
    end = timer()
    print(f"total training time: {timedelta(seconds=end-start)}")


def main():
    seed = 43
    setup_seed(seed)
    
    # train one epoch on balanced data
    train_df = pd.read_csv("./data/train_balanced_df.csv")
    num_positives = np.sum(train_df.label)
    num_negatives = len(train_df.label) - num_positives
    pos_weight  = num_negatives / num_positives
    print(f"pos weight: {pos_weight}")
    tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')
    train_ds = BertDataset(train_df, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True)
    model = EndToEndClassifier()
    train(model, train_dataloader, pos_weight=pos_weight, epochs=1, seed=seed)

    # train one epoch on less skewed data
    train_df = pd.read_csv("./data/train_less_skewed_df.csv")
    num_positives = np.sum(train_df.label)
    num_negatives = len(train_df.label) - num_positives
    pos_weight  = num_negatives / num_positives
    print(f"pos weight: {pos_weight}")
    train_ds = BertDataset(train_df, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True)
    train(model, train_dataloader, pos_weight=pos_weight, epochs=1, start_epoch=1, seed=seed, bert_lr=1e-5)
    
    # train one epoch on full dataset
    train_df = pd.read_csv("./data/train_df.csv")
    num_positives = np.sum(train_df.label)
    num_negatives = len(train_df.label) - num_positives
    pos_weight  = num_negatives / num_positives
    print(f"pos weight: {pos_weight}")
    train_ds = BertDataset(train_df, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True)
    train(model, train_dataloader, pos_weight=pos_weight, epochs=1, start_epoch=2, seed=seed, bert_lr=5e-6, linear_lr=0.0001)

main()
