#!/usr/bin/env python3
import hydra
import numpy as np
import tritonclient.http as httpclient
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="simple", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    x = np.random.random((cfg.batch_size, 3, cfg.height, cfg.width)).astype(np.float32)
    client = httpclient.InferenceServerClient(url=cfg.url)
    infer_input = httpclient.InferInput("input", x.shape, "FP32")
    infer_input.set_data_from_numpy(x)
    requested_output = httpclient.InferRequestedOutput("logits")

    result = client.infer(
        model_name=cfg.model_name,
        inputs=[infer_input],
        outputs=[requested_output],
    )
    logits = result.as_numpy("logits")
    if logits is None:
        raise RuntimeError("Triton response did not contain output 'logits'")

    print(f"model name: {cfg.model_name}")
    print(f"input shape: {x.shape}")
    print(f"output shape: {logits.shape}")
    print(f"output dtype: {logits.dtype}")
    print(f"output min: {logits.min():.6f}")
    print(f"output max: {logits.max():.6f}")
    print(f"output mean: {logits.mean():.6f}")


if __name__ == "__main__":
    main()
