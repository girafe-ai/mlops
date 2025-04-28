import numpy as np
import torch
import pytorch_lightning as pl
import onnxruntime as ort

from pl_modules.model import ImageClassifier

class InferenceModel(pl.LightningModule):
    def __init__(self, model_path: str):
        super().__init__()
        self.model = ImageClassifier.load_from_checkpoint(model_path)
    
    def forward(self, x):
        out = self.model(x)
        return out


def main():
    model = InferenceModel("../models/epoch=04-val_loss=0.5945.ckpt")
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dummy_input = torch.randn(1, 3, 96, 96, device=device)

    torch.onnx.export(
        model, 
        dummy_input, 
        "conv_model.onnx", 
        export_params=True,
        input_names=['PREPROCESSED_IMAGE'], 
        output_names=['LOGITS'],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            'PREPROCESSED_IMAGE': {0: 'batch_size'},
            'LOGITS': {0: 'batch_size'}
        }
    )

def check_onnx(onnx_model_path: str):
    inputs = torch.randn(64, 3, 96, 96)
    ort_sess = ort.InferenceSession(onnx_model_path)
    outputs = ort_sess.run(None, {'PREPROCESSED_IMAGE': inputs.numpy().astype(np.float32)})

    print(outputs)
    print(outputs[0].shape)


if __name__ == "__main__":
    check_onnx("conv_model.onnx")
    # main()