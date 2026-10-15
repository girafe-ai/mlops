import pytest
from hydra import compose, initialize_config_module


@pytest.fixture(params=[255, -1], ids=["default-ignore", "negative-ignore"])
def data_config(request):
    with initialize_config_module(
        version_base=None, config_module="cityscapes_tools.conf"
    ):
        return compose(
            config_name="torch",
            overrides=[f"data.ignore_index={request.param}"],
        ).data
