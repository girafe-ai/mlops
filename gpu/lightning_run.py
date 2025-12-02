from data import QwenDataModule
from model import QwenModule
import lightning as L

dm = QwenDataModule()
module = QwenModule("Qwen/Qwen3-0.6B")
trainer = L.Trainer(max_epochs=1, accelerator="gpu", devices=[0, 1])
trainer.fit(module, dm)
