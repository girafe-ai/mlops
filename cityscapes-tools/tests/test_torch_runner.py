from unittest.mock import Mock

from cityscapes_tools.training.torch import runner
from hydra import compose, initialize_config_module


def test_torch_runner_passes_configured_ignore_index(monkeypatch, data_config) -> None:
    with initialize_config_module(
        version_base=None, config_module="cityscapes_tools.conf"
    ):
        config = compose(config_name="torch")
    config.data = data_config
    config.training.max_epochs = 0
    config.training.device = "cpu"
    dataset_factory = Mock()
    loader_factory = Mock()
    loss_factory = Mock(wraps=runner.torch.nn.CrossEntropyLoss)
    train_epoch = Mock(side_effect=AssertionError("Training must not run"))
    validate_epoch = Mock(side_effect=AssertionError("Validation must not run"))
    monkeypatch.setattr(runner, "create_dataset", dataset_factory)
    monkeypatch.setattr(runner, "create_dataloader", loader_factory)
    monkeypatch.setattr(runner.torch.nn, "CrossEntropyLoss", loss_factory)
    monkeypatch.setattr(runner, "train_one_epoch", train_epoch)
    monkeypatch.setattr(runner, "validate", validate_epoch)

    runner.run(config)

    assert dataset_factory.call_count == 2
    assert loader_factory.call_count == 2
    for factory in (dataset_factory, loader_factory, loss_factory):
        for call in factory.call_args_list:
            assert call.kwargs["ignore_index"] == data_config.ignore_index
    train_epoch.assert_not_called()
    validate_epoch.assert_not_called()
