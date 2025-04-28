import os

import pytest
from cats_and_dogs.pl_modules.model import ImageClassifier


@pytest.mark.requires_files
def test_model_loading(request):
    model_path = request.config.getoption("--model-path")
    if not model_path:
        pytest.skip("Use --model-path=<path>")

    os.system(f"dvc pull {model_path}")
    model = ImageClassifier.load_from_checkpoint(model_path.rsplit(".", 1)[0])
    assert model is not None, "Model should not be None!"
