import numpy as np
import torch

from cats_and_dogs.data import init_dataloader, init_dataset
from cats_and_dogs.model import SimpleClassifier
from cats_and_dogs.types import Directory, File


@torch.no_grad()
def evaluate(model, test_loader, device, subset="test"):
    """Inference of the model

    Args:
        model: model to infer
        test_loader: dataloader for test data
        device: device used for inference
        subset: just for prettier printing. Defaults to "test".
    """
    model.train(False)
    test_batch_acc = []

    print("Start testing...")
    for x_batch, y_batch in test_loader:
        logits = model(x_batch.to(device))
        y_pred = logits.max(1)[1].data
        test_batch_acc.append(np.mean((y_batch.cpu() == y_pred.cpu()).numpy()))

    test_accuracy = np.mean(test_batch_acc)

    print("Results:")
    print(f"    {subset} accuracy: {test_accuracy * 100:.2f} %")


def infer(model_file: File, data_dir: Directory):
    model = SimpleClassifier()

    checkpoint = torch.load(model_file, weights_only=True)
    model.load_state_dict(checkpoint)

    test_dataset = init_dataset(data_dir)
    test_loader = init_dataloader(test_dataset, 128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate(model, test_loader, device)
