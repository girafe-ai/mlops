import torch

from cats_and_dogs.pl_modules.classifiers import SimpleClassifier


def test_model_prediction():
    model = SimpleClassifier(num_classes=2)
    dummy_input = torch.rand(1, 96, 96, 3)

    prediction = model(dummy_input)
    assert prediction.shape == (1, 2), "Output should be (batch_size, num_classes)!"
    assert torch.allclose(
        prediction.sum(), torch.tensor(1.0)
    ), "Predictions should sum to 1 (softmax)!"
