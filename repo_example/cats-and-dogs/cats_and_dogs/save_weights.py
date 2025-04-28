import torch
import pytorch_lightning as pl

from pl_modules.model import ImageClassifier

def main():
    model = ImageClassifier.load_from_checkpoint("../models/epoch=04-val_loss=0.5945.ckpt")
    torch.save(model.model.state_dict(), "model_weights.ckpt")

if __name__ == "__main__":
    main()
