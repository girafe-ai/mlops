import torch
import torchvision.models as models
import onnxruntime as ort
import numpy as np


def main():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        "resnet50.onnx",
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=["IMAGES"],
        output_names=["CLASS_PROBS"],
        dynamic_axes={"IMAGES": {0: "BATCH_SIZE"}, "CLASS_PROBS": {0: "BATCH_SIZE"}},
    )

    # Comparing ort and torch outputs
    original_embeddings = model(dummy_input).detach().numpy()
    ort_inputs = {
        "IMAGES": dummy_input.numpy(),
    }
    ort_session = ort.InferenceSession("resnet50.onnx")
    onnx_embeddings = ort_session.run(None, ort_inputs)[0]

    assert np.allclose(original_embeddings, onnx_embeddings, atol=1e-5)


if __name__ == "__main__":
    main()
