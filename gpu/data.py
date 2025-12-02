import lightning as L
from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import torch


class QwenDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        load_dataset("HuggingFaceH4/MATH-500")
        AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")

    def setup(self, stage: str):
        if stage == "fit":
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")

            def apply_tokenizer(item):
                tokens = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": item["problem"]},
                        {"role": "assistant", "content": item["solution"]},
                    ],
                    tokenize=True,
                    enable_thinking=False,
                    return_tensors="pt",
                )[0]
                return {
                    "input_ids": tokens,
                    "labels": torch.cat([tokens[1:], torch.tensor([-100])]),
                }

            dataset = (
                load_dataset("HuggingFaceH4/MATH-500")["test"]
                .map(apply_tokenizer)
                .with_format("torch")
                .select_columns(["input_ids", "labels"])
                .shuffle(seed=42)
                .train_test_split(test_size=0.1)
            )
            self.train_dataset = dataset["train"]
            self.val_dataset = dataset["test"]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=1, shuffle=True, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=1, shuffle=False, pin_memory=True
        )
