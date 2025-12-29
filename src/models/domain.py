"""領域適應模塊

提供領域對抗訓練所需的梯度反轉層和領域分類器
用於域適應任務，幫助模型學習域不變特征
"""
import torch
from torch import nn


class GradientReversal(torch.autograd.Function):
    """梯度反轉層

    在正向傳播時保持輸入不變，在反向傳播時反轉梯度方向
    用於領域對抗訓練中的梯度流控制
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        """正向傳播

        Args:
            ctx: 上下文對象，用於保存反向傳播所需的參數
            x: 輸入特征
            lambda_: 梯度反轉係數

        Returns:
            與輸入相同的特征
        """
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """反向傳播

        Args:
            ctx: 上下文對象，包含正向傳播時保存的參數
            grad_output: 來自下一層的梯度

        Returns:
            反轉後的梯度和None（lambda_的梯度）
        """
        return grad_output.neg() * ctx.lambda_, None


class DomainClassifier(nn.Module):
    """領域分類器

    用於判斷輸入特征來自哪個數據域
    與梯度反轉層結合使用，實現領域對抗訓練

    Args:
        input_dim: 輸入特征維度
        hidden_dim: 隱藏層維度
        num_domains: 領域數量
        dropout: Dropout概率
    """
    def __init__(self, input_dim, hidden_dim=128, num_domains=2, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),  # 層歸一化
            nn.Linear(input_dim, hidden_dim),  # 線性層
            nn.ReLU(),  # 激活函數
            nn.Dropout(dropout),  # Dropout正則化
            nn.Linear(hidden_dim, num_domains),  # 領域分類輸出層
        )

    def forward(self, features, grl_lambda=1.0):
        """前向傳播

        Args:
            features: 輸入特征，形狀為 (batch_size, input_dim)
            grl_lambda: 梯度反轉係數

        Returns:
            領域分類預測，形狀為 (batch_size, num_domains)
        """
        # 應用梯度反轉層
        reversed_features = GradientReversal.apply(features, grl_lambda)
        # 領域分類
        return self.net(reversed_features)
