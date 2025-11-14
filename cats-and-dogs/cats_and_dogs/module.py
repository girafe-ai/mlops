import lightning as L
import torch.nn
import torchmetrics


class CatsAndDogsModule(L.LightningModule):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.criterion = torch.nn.BCELoss()
        self.val_accuracy = torchmetrics.Accuracy("binary")
        self.test_accuracy = torchmetrics.Accuracy("binary")

    def forward(self, inputs):
        return self.model(inputs)[:, 1]

    def training_step(self, batch):
        inputs, target = batch
        y_proba = self.forward(inputs)
        loss = self.criterion(y_proba, target.to(torch.float))
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch):
        inputs, target = batch
        y_proba = self.forward(inputs)
        loss = self.criterion(y_proba, target.to(torch.float))
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        y_predict = (y_proba > 0.5).to(torch.long)
        self.val_accuracy(y_predict, target)
        self.log(
            "val_accuracy",
            self.val_accuracy,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch):
        inputs, target = batch
        y_proba = self.forward(inputs)
        loss = self.criterion(y_proba, target.to(torch.float))
        self.log("test_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        y_predict = (y_proba > 0.5).to(torch.long)
        self.test_accuracy(y_predict, target)
        self.log(
            "test_accuracy",
            self.test_accuracy,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())
