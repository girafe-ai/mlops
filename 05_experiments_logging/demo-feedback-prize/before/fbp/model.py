import torch
import transformers


class MyModel(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        num_target_cols: int,
        dropout: float,
        freeze_backbone: bool,
    ):
        super().__init__()
        self.backbone = transformers.AutoModel.from_pretrained(
            model_name,
            return_dict=True,
            output_hidden_states=True,
        )
        self.drop = torch.nn.Dropout(p=dropout)
        self.fc = torch.nn.LazyLinear(out_features=num_target_cols)
        self.freeze_backbone = freeze_backbone

    def forward(self, input_ids, attention_mask):
        if self.freeze_backbone:
            with torch.no_grad():
                embeddings = self.backbone(input_ids, attention_mask)[
                    "last_hidden_state"
                ][:, 0]
        else:
            embeddings = self.backbone(input_ids, attention_mask)["last_hidden_state"][
                :, 0
            ]

        embeddings = self.drop(embeddings)
        logits = self.fc(embeddings)
        return logits
