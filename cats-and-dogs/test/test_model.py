import torch
import pytest

from cats_and_dogs.model import SimpleClassifier, ConvClassifier


def test_simple_architecture():
    model = SimpleClassifier()
    assert len(model.model) == 7


@pytest.fixture
def simple_model():
    return SimpleClassifier()


@pytest.fixture
def conv_model():
    return ConvClassifier(num_classes=2)


@pytest.mark.parametrize("model_type", ["simple", "conv"])
@pytest.mark.parametrize("batch_size", [16, 32, 64])
def test_simple_forward(batch_size: int, model_type: str):
    if model_type == "simple":
        model = SimpleClassifier()
    elif model_type == "conv":
        model = ConvClassifier(2)

    batch = torch.zeros((batch_size, 3, 96, 96))
    predict = model(batch)
    assert predict.shape[0] == batch_size
    assert predict.shape[1] == 2


def test_model_load(request):
    model_path = request.config.getoption("--model-path")
    if model_path is None:
        pytest.skip("No model provided")
    model = SimpleClassifier()
    model.load_state_dict(torch.load(model_path))
