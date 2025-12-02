from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist


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
    train_sampler = DistributedSampler(
        train_dataset, dist.get_world_size(), dist.get_rank(), shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, dist.get_world_size(), dist.get_rank(), shuffle=False
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=1, sampler=train_sampler, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, sampler=val_sampler, pin_memory=True
    )
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
    torch.cuda.set_device(dist.get_rank())
    device = torch.device("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

    model.to(device)
    model = DistributedDataParallel(model)
    model.train()
    for i, batch in enumerate(train_dataloader):
        # if i == 10:
        # break
        optimizer.zero_grad()
        batch["input_ids"] = batch["input_ids"].to(device)
        batch["labels"] = batch["labels"].to(device)
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        if dist.get_rank() == 0:
            print(f"Loss: {loss.item()}")
    val_loss = []
    model.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            batch["input_ids"] = batch["input_ids"].to(device)
            batch["labels"] = batch["labels"].to(device)
            val_loss_item = model(**batch).loss
            dist.all_reduce(val_loss_item)
            val_loss.append(val_loss_item.item())

    if dist.get_rank() == 0:
        print(f"Val loss: {sum(val_loss) / len(val_loss)}")


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    main()
    dist.destroy_process_group()
