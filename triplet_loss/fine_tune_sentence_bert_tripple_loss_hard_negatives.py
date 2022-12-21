# -*- coding: utf-8 -*-
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import random
from sentence_transformers.readers import InputExample
from sentence_transformers import SentenceTransformer, LoggingHandler, losses


class TripletDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]
        anchor = item["ChallengeDescription"]
        positive = item["abstract"]
        negative = item["abstract_negative"]
        return InputExample(texts=[anchor, positive, negative])


def train(seed, train_df, batch_size=8, epochs=3, lr=3e-5):
    train_ds = TripletDataset(train_df)
    print(f"steps per epoch: {len(train_ds)//batch_size}")
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    # load checkpoint trained on warm up dataset
    model = SentenceTransformer(f"./data/checkpoints/{seed}/10606")
    train_loss = losses.TripletLoss(model=model)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        checkpoint_save_steps=len(train_ds) // batch_size,
        epochs=epochs,
        warmup_steps=1000,
        optimizer_params={"lr": lr},
        checkpoint_path=f"./data/checkpoints_hard_negatives/{str(seed)}",
    )
    return model


def main():
    seed = 43
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_df = pd.read_csv("./data/hard_negatives.csv")
    model = train(seed, train_df)
    model.save(f"./data/final_model_hard_negatives/{seed}")


main()
