import torch
from torch import nn


class PromptTokens(nn.Module):
    def __init__(self, n_prompt, d_model):
        super().__init__()
        self.n_prompt = n_prompt
        if n_prompt > 0:
            self.embeddings = nn.Parameter(torch.randn(n_prompt, d_model) * 0.02)
        else:
            self.embeddings = None

    def forward(self, x, lengths):
        if self.n_prompt <= 0:
            padding_mask = build_padding_mask(x.size(0), x.size(1), lengths, x.device)
            return x, padding_mask, 0

        batch_size = x.size(0)
        prompt = self.embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.cat([prompt, x], dim=1)

        prompt_mask = torch.zeros(
            batch_size, self.n_prompt, dtype=torch.bool, device=x.device
        )
        frame_mask = build_padding_mask(
            batch_size, x.size(1) - self.n_prompt, lengths, x.device
        )
        padding_mask = torch.cat([prompt_mask, frame_mask], dim=1)
        return x, padding_mask, self.n_prompt


def build_padding_mask(batch_size, max_len, lengths, device):
    steps = torch.arange(max_len, device=device).unsqueeze(0)
    lengths = lengths.unsqueeze(1)
    return steps >= lengths
