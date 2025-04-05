def summary_grids():
    return {
        "data_loading": {
            "train_data_path": "data/train_11k",
            "val_data_path": "data/val",
            "test_data_path": "data/test_labeled",
        },
        "training": {"batch_size": 2, "num_workers": 1},
        "model": {
            "image_height": 32,
            "image_width": 32,
            "image_mean": [0.0, 0.0, 0.0],
            "image_std": [0.1, 0.1, 0.1],
        },
    }


def test_config(data_regression):
    data = summary_grids()
    data_regression.check(data)
