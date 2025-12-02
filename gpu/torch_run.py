from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader


def get_datasets():
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
    return dataset["train"], dataset["test"]


def main():
    train_dataset, val_dataset = get_datasets()
    train_dataloader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, pin_memory=True
    )
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")
    device = torch.device("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

    model.to(device)
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        batch["input_ids"] = batch["input_ids"].to(device)
        batch["labels"] = batch["labels"].to(device)
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")
    val_loss = []
    model.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            batch["input_ids"] = batch["input_ids"].to(device)
            batch["labels"] = batch["labels"].to(device)
            val_loss.append(model(**batch).loss.item())
    print(f"Val loss: {sum(val_loss) / len(val_loss)}")


if __name__ == "__main__":
    main()
