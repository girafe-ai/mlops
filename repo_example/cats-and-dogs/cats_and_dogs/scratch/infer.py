import numpy as np
import torch

from cats_and_dogs.scratch.data import init_dataloader, init_dataset
from cats_and_dogs.scratch.model import SimpleClassifier


@torch.no_grad()
def infer_model(model, test_loader, device, subset="test"):
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
    for X_batch, y_batch in test_loader:
        logits = model(X_batch.to(device))
        y_pred = logits.max(1)[1].data
        test_batch_acc.append(np.mean((y_batch.cpu() == y_pred.cpu()).numpy()))

    test_accuracy = np.mean(test_batch_acc)

    print("Results:")
    print(f"    {subset} accuracy: {test_accuracy * 100:.2f} %")


def main():
    model = SimpleClassifier()

    checkpoint = torch.load("../models/simple_model_0.55.pt", weights_only=True)
    model.load_state_dict(checkpoint)

    test_dataset = init_dataset("../data/test_labeled")
    test_loader = init_dataloader(test_dataset, 128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    infer_model(model, test_loader, device)


if __name__ == "__main__":
    main()
