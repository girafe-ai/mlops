"""Benchmark inference across PyTorch and ONNX Runtime backends."""

import hydra
import pandas as pd
import torch
from inference_seminar.backends import (
    BackendResult,
    ort_session_for_backend,
    run_ort_session,
    run_pytorch_model,
)
from inference_seminar.benchmarking import measure_callable
from inference_seminar.cityscapes_adapter import (
    build_preprocess_transform,
    collect_image_paths,
    load_cityscapes_model,
    load_images_as_tensor_batch,
    resolve_device,
    resolve_repo_path,
)
from inference_seminar.exporting import export_cityscapes_to_onnx
from omegaconf import DictConfig


def benchmark_pytorch(
    model: torch.nn.Module,
    batch: torch.Tensor,
    warmup_runs: int,
    timed_runs: int,
) -> tuple[BackendResult, object]:
    device = batch.device
    stats = measure_callable(
        lambda: model(batch),
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        batch_size=batch.shape[0],
        device=device,
    )
    output = run_pytorch_model(model, batch)
    return (
        BackendResult(
            backend="pytorch",
            available=True,
            provider="torch",
            mean_ms=stats.mean_ms,
            median_ms=stats.median_ms,
            p95_ms=stats.p95_ms,
            throughput_items_per_s=stats.throughput_items_per_s,
            max_abs_diff_vs_pytorch=0.0,
        ),
        output,
    )


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    artifacts_dir = resolve_repo_path(cfg.paths.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    resolve_repo_path(cfg.benchmark.trt_engine_cache_path).mkdir(
        parents=True, exist_ok=True
    )
    onnx_path = resolve_repo_path(cfg.benchmark.onnx_path)
    if not onnx_path.exists():
        exported_path, _, max_abs_diff = export_cityscapes_to_onnx(cfg)
        print(
            "Exported ONNX model automatically before benchmarking: "
            f"{exported_path} (max abs diff {max_abs_diff:.6e})"
        )

    device = resolve_device(cfg.model.device)
    model = load_cityscapes_model(
        cfg.paths.cityscapes_project_dir,
        cfg.paths.cityscapes_checkpoint,
        device,
    )
    transform = build_preprocess_transform(
        cfg.paths.cityscapes_project_dir,
        cfg.model.height,
        cfg.model.width,
    )
    image_paths = collect_image_paths(
        cfg.paths.cityscapes_input_dir,
        limit=cfg.benchmark.num_images,
    )
    batch_cpu, image_names = load_images_as_tensor_batch(image_paths, transform)
    batch = batch_cpu.to(device)

    results: list[BackendResult] = []
    pytorch_result, reference_output = benchmark_pytorch(
        model,
        batch,
        warmup_runs=int(cfg.benchmark.warmup_runs),
        timed_runs=int(cfg.benchmark.timed_runs),
    )
    results.append(pytorch_result)

    ort_input = batch_cpu.numpy()
    for backend_name in cfg.benchmark.providers:
        if backend_name == "pytorch":
            continue

        session, provider_name, notes = ort_session_for_backend(
            backend_name,
            onnx_path,
            trt_engine_cache_path=resolve_repo_path(
                cfg.benchmark.trt_engine_cache_path
            ),
        )
        if session is None:
            results.append(
                BackendResult(
                    backend=backend_name,
                    available=False,
                    provider=provider_name,
                    mean_ms=None,
                    median_ms=None,
                    p95_ms=None,
                    throughput_items_per_s=None,
                    max_abs_diff_vs_pytorch=None,
                    notes=notes,
                )
            )
            continue

        stats = measure_callable(
            lambda: run_ort_session(session, ort_input),
            warmup_runs=int(cfg.benchmark.warmup_runs),
            timed_runs=int(cfg.benchmark.timed_runs),
            batch_size=ort_input.shape[0],
            device=torch.device("cuda")
            if provider_name != "CPUExecutionProvider"
            else None,
        )
        output = run_ort_session(session, ort_input)
        max_abs_diff = float(abs(output - reference_output).max())
        results.append(
            BackendResult(
                backend=backend_name,
                available=True,
                provider=provider_name,
                mean_ms=stats.mean_ms,
                median_ms=stats.median_ms,
                p95_ms=stats.p95_ms,
                throughput_items_per_s=stats.throughput_items_per_s,
                max_abs_diff_vs_pytorch=max_abs_diff,
                notes=notes,
            )
        )

    df = pd.DataFrame([result.__dict__ for result in results])
    df["num_images"] = len(image_names)
    df["height"] = cfg.model.height
    df["width"] = cfg.model.width
    output_path = resolve_repo_path(cfg.benchmark.results_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved benchmark results to: {output_path}")


if __name__ == "__main__":
    main()
