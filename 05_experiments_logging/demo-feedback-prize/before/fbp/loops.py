from typing import Optional

from tqdm import tqdm
import torch


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    grad_accum_steps: int,
    use_amp: bool,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> float:
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    progress_bar = tqdm(data_loader, desc="Training", dynamic_ncols=True)
    for step, (input_ids, attention_mask, labels) in enumerate(progress_bar):
        input_ids, attention_mask, labels = (
            input_ids.to(device),
            attention_mask.to(device),
            labels.to(device),
        )

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits.squeeze(), labels.squeeze())

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (step + 1) % grad_accum_steps == 0 or step == len(data_loader) - 1:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
        
        running_loss += loss.item()
        progress_bar.set_postfix({"train_loss": f"{running_loss/(step+1):.2f}"})
        
    progress_bar.close()
    epoch_loss = running_loss / len(data_loader)
    return epoch_loss


def validate_one_epoch(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> float:
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(data_loader, desc="Validating"):
            input_ids, attention_mask, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                labels.to(device),
            )

            with torch.cuda.amp.autocast(use_amp):
                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits.squeeze(), labels.squeeze())

            running_loss += loss.item()

    epoch_loss = running_loss / len(data_loader)
    return epoch_loss


def train_loop(
    model,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    num_epochs: int,
    use_amp: bool,
    grad_accum_steps: int,
    loss_fn,
):
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            loss_fn=loss_fn,
            device=device,
            grad_accum_steps=grad_accum_steps,
            use_amp=use_amp,
            scheduler=scheduler,
            scaler=scaler,
        )

        val_loss = validate_one_epoch(
            model=model,
            data_loader=val_dataloader,
            loss_fn=loss_fn,
            device=device,
            use_amp=use_amp,
        )

        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

        print("-" * 60)
