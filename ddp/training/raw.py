import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import MNIST


def init_process(local_rank, fn, backend="nccl"):
    """Initialize the distributed environment."""
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    fn(local_rank, size)


torch.set_num_threads(1)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4608, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def run_training(rank, size):
    torch.manual_seed(1234)
    dataset = MNIST(
        "./mnist",
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    loader = DataLoader(
        dataset, sampler=DistributedSampler(dataset, size, rank), batch_size=16
    )
    model = Net()
    device = torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = len(loader)
    steps = 0
    epoch_loss = 0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        epoch_loss += loss.item()
        loss.backward()
        average_gradients(model)
        optimizer.step()
        steps += 1
        if True:
            print(f"Rank {dist.get_rank()}, loss: {epoch_loss / num_batches}")
            epoch_loss = 0


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_process(local_rank, fn=run_training, backend="gloo")
