import lightning as L
from torch.optim import AdamW
from transformers import AutoModelForCausalLM
import torch

torch.set_float32_matmul_precision("highest")


class QwenModule(L.LightningModule):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def forward(self, batch):
        output = self.model(**batch)
        return output.loss

    def training_step(self, batch):
        loss = self.forward(batch)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch):
        loss = self.forward(batch)
        self.log("val/loss", loss)

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=1e-6)
