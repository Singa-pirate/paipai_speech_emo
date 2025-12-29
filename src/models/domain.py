import torch
from torch import nn


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class DomainClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_domains=2, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, features, grl_lambda=1.0):
        reversed_features = GradientReversal.apply(features, grl_lambda)
        return self.net(reversed_features)
