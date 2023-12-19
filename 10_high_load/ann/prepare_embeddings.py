from typing import Any
import json

import numpy as np
import pandas as pd
import torch
import open_clip
import lightning.pytorch as pl


def load_dataframe(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        data_full = json.load(f)

    dataset_list = [
        pd.DataFrame(data_full[key], columns=["intents", "targets"])
        for key in data_full.keys()
    ]
    total_df = pd.concat(dataset_list, axis=0, sort=False).reset_index(drop=True)
    return total_df


class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: Any, df: pd.DataFrame):
        super().__init__()
        self.tokenizer = tokenizer
        self.df = df.values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        intent, target = self.df[idx]
        token_ids = self.tokenizer(intent).squeeze(0)
        return token_ids, intent, target


class InferenceModule(pl.LightningModule):
    def __init__(self, text_model: torch.nn.Module):
        super().__init__()
        self.text_model = text_model

        self.random_proj_128 = torch.nn.LazyLinear(out_features=128)
        self.random_proj_64 = torch.nn.LazyLinear(out_features=64)
        self.random_proj_32 = torch.nn.LazyLinear(out_features=32)
        self.random_proj_16 = torch.nn.LazyLinear(out_features=16)
        self.random_proj_8 = torch.nn.LazyLinear(out_features=8)

    def forward(self, token_ids):
        embeddings = self.text_model(token_ids)
        embeddings_128 = torch.nn.functional.normalize(self.random_proj_128(embeddings))
        embeddings_64 = torch.nn.functional.normalize(self.random_proj_64(embeddings))
        embeddings_32 = torch.nn.functional.normalize(self.random_proj_32(embeddings))
        embeddings_16 = torch.nn.functional.normalize(self.random_proj_16(embeddings))
        embeddings_8 = torch.nn.functional.normalize(self.random_proj_8(embeddings))
        embeddings = torch.nn.functional.normalize(embeddings)

        return (
            embeddings,
            embeddings_128,
            embeddings_64,
            embeddings_32,
            embeddings_16,
            embeddings_8,
        )

    def predict_step(self, batch):
        token_ids, intents, targets = batch
        to_numpy = lambda x: x.detach().cpu().numpy().astype(np.float32)

        (
            embeddings,
            embeddings_128,
            embeddings_64,
            embeddings_32,
            embeddings_16,
            embeddings_8,
        ) = self(token_ids)
        embeddings = to_numpy(embeddings)
        embeddings_128 = to_numpy(embeddings_128)
        embeddings_64 = to_numpy(embeddings_64)
        embeddings_32 = to_numpy(embeddings_32)
        embeddings_16 = to_numpy(embeddings_16)
        embeddings_8 = to_numpy(embeddings_8)

        return (
            embeddings,
            embeddings_128,
            embeddings_64,
            embeddings_32,
            embeddings_16,
            embeddings_8,
            intents,
            targets,
        )


def process_data(
    tokenizer: Any,
    model: torch.nn.Module,
    df: pd.DataFrame,
    output_file: str,
    batch_size: int = 512,
    num_workers: int = 16,
):
    dataset = Dataset(tokenizer, df)
    pl_module = InferenceModule(model)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    trainer = pl.Trainer(accelerator="cuda", devices=[3], precision="16-mixed")
    results = trainer.predict(pl_module, dataloader)

    total = []
    for batch in results:
        total.extend(zip(*batch))

    pd.DataFrame(
        total,
        columns=[
            "emb",
            "proj128",
            "proj64",
            "proj32",
            "proj16",
            "proj8",
            "intent",
            "target",
        ],
    ).to_parquet(output_file, index=False)


def main():
    df_real = load_dataframe("./oos-eval/data/data_full.json")
    df_synthetic = pd.DataFrame(
        [(str(i), "synt") for i in range(1_000_000)], columns=["intent", "target"]
    )

    model, _, _ = open_clip.create_model_and_transforms(
        "xlm-roberta-base-ViT-B-32",
        pretrained="laion5b_s13b_b90k",
        force_custom_text=True,
    )
    tokenizer = open_clip.get_tokenizer("xlm-roberta-base-ViT-B-32")

    process_data(tokenizer, model.text, df_real, "xlm-roberta-embeddings.parquet")
    process_data(
        tokenizer, model.text, df_synthetic, "xlm-roberta-synthetic-embeddings.parquet"
    )


if __name__ == "__main__":
    main()
