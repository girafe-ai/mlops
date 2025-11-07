import torch

from cats_and_dogs.data import init_dataset, init_dataloader
from cats_and_dogs.infer import evaluate
from cats_and_dogs.types import Directory, File
from cats_and_dogs.model import SimpleClassifier

def train(train_data_dir: Directory, val_data_dir: Directory, output_file: File, batch_size: int = 32, num_epochs: int = 1):
    model = SimpleClassifier()

    train_dataset = init_dataset(train_data_dir)
    train_loader = init_dataloader(train_dataset, batch_size)

    val_dataset = init_dataset(val_data_dir)
    val_loader = init_dataloader(val_dataset, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()

    for epoch in range(num_epochs):
        train_loss = 0.0
        num_train_steps = 0
        for batch in train_loader:
            optimizer.zero_grad()
            y_proba = model(batch[0].to(device))[:, 1]
            loss = criterion(y_proba, batch[1].to(torch.float))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_train_steps += 1

        val_loss = 0.0
        val_accuracy = 0.0
        num_val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                y_proba = model(batch[0].to(device))[:, 1]
                y_pred = (y_proba > 0.5).to(torch.long)
                loss = criterion(y_proba, batch[1].to(torch.float))
                val_loss += loss.item()
                val_accuracy += (y_pred == batch[1]).sum()
                num_val_steps += 1

        print(f"Epoch={epoch}, train loss = {train_loss / num_train_steps}, val loss = {val_loss / num_val_steps}, val accuracy = {val_accuracy / num_val_steps}")
    
    torch.save(model.state_dict(), output_file)
