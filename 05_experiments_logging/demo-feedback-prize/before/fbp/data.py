from typing import List, Tuple

import torch
import pandas as pd
import transformers
from sklearn.model_selection import train_test_split


class MyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        text_max_length: int,
        labels: List[str],
    ):
        super().__init__()
        self.text_max_length = text_max_length
        self.texts = list(dataframe["full_text"])
        self.labels = dataframe[labels].values
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text = self.texts[idx]
        inputs = self.tokenizer(
            text,
            None,
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=True,
            max_length=self.text_max_length,
            padding="max_length",
        )
        input_ids = torch.LongTensor(inputs["input_ids"], device="cpu")
        attention_mask = torch.LongTensor(inputs["attention_mask"], device="cpu")
        labels = torch.FloatTensor([list(self.labels[idx])], device="cpu")
        return input_ids, attention_mask, labels


def get_dataloders(
    csv_path: str,
    val_size: float,
    dataloader_num_wokers: int,
    batch_size: int,
    tokenizer_model_name: str,
    text_max_length: int,
    labels: List[str],
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_model_name)
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=val_size)
    train_dataloader = torch.utils.data.DataLoader(
        dataset=MyDataset(
            dataframe=train_df,
            tokenizer=tokenizer,
            text_max_length=text_max_length,
            labels=labels,
        ),
        batch_size=batch_size,
        num_workers=dataloader_num_wokers,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=MyDataset(
            dataframe=val_df,
            tokenizer=tokenizer,
            text_max_length=text_max_length,
            labels=labels,
        ),
        batch_size=batch_size,
        num_workers=dataloader_num_wokers,
    )
    return train_dataloader, val_dataloader
