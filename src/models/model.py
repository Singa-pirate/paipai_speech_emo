from torch import nn

from .backbone import TransformerBackbone, build_sinusoidal_positions
from .head import ClassificationHead
from .prompt import PromptTokens


def mean_pool(features, padding_mask):
    if padding_mask is None:
        return features.mean(dim=1)
    valid = (~padding_mask).float().unsqueeze(-1)
    summed = (features * valid).sum(dim=1)
    denom = valid.sum(dim=1).clamp(min=1.0)
    return summed / denom


class EmotionModel(nn.Module):
    def __init__(
        self,
        n_mels,
        num_classes,
        d_model=256,
        n_heads=4,
        n_layers=4,
        ff_mult=4,
        dropout=0.1,
        n_prompt=8,
        pool="mean",
    ):
        super().__init__()
        self.pool = pool
        self.prompt = PromptTokens(n_prompt, d_model)
        self.backbone = TransformerBackbone(
            input_dim=n_mels,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_mult=ff_mult,
            dropout=dropout,
        )
        self.head = ClassificationHead(d_model, num_classes, dropout=dropout)
        self.embedding_dim = d_model

    def forward(self, features, lengths):
        x = self.backbone.input_proj(features)
        pos = build_sinusoidal_positions(x.size(1), x.size(2), x.device)
        x = x + pos.unsqueeze(0)
        x, padding_mask, prompt_len = self.prompt(x, lengths)
        x = self.backbone.encoder(
            self.backbone.dropout(x),
            src_key_padding_mask=padding_mask,
        )

        if prompt_len > 0:
            frame_x = x[:, prompt_len:, :]
            frame_mask = padding_mask[:, prompt_len:]
        else:
            frame_x = x
            frame_mask = padding_mask

        if self.pool == "cls" and prompt_len > 0:
            pooled = x[:, 0, :]
        else:
            pooled = mean_pool(frame_x, frame_mask)
        logits = self.head(pooled)
        return logits, pooled

    def get_param_groups(self):
        encoder_params = list(self.backbone.parameters())
        head_params = list(self.head.parameters())
        if self.prompt.embeddings is not None:
            head_params.append(self.prompt.embeddings)
        return encoder_params, head_params
