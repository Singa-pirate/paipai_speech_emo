"""提示令牌模塊

提供提示令牌的實現，用於在Transformer中引入可學習的提示
增強模型的上下文理解能力
"""
import torch
from torch import nn


class PromptTokens(nn.Module):
    """提示令牌

    在輸入序列前添加可學習的提示令牌，用於增強模型表示

    Args:
        n_prompt: 提示令牌的數量
        d_model: 模型嵌入維度
    """
    def __init__(self, n_prompt, d_model):
        super().__init__()
        self.n_prompt = n_prompt
        # 初始化提示令牌嵌入
        if n_prompt > 0:
            self.embeddings = nn.Parameter(torch.randn(n_prompt, d_model) * 0.02)
        else:
            self.embeddings = None

    def forward(self, x, lengths):
        """前向傳播

        Args:
            x: 輸入特征，形狀為 (batch_size, seq_length, d_model)
            lengths: 每個樣本的有效序列長度，形狀為 (batch_size,)

        Returns:
            x: 添加提示令牌後的特征，形狀為 (batch_size, seq_length + n_prompt, d_model)
            padding_mask: 填充掩碼，形狀為 (batch_size, seq_length + n_prompt)
            prompt_len: 提示令牌的長度
        """
        if self.n_prompt <= 0:
            padding_mask = build_padding_mask(x.size(0), x.size(1), lengths, x.device)
            return x, padding_mask, 0

        batch_size = x.size(0)
        # 將提示令牌擴展到批次大小
        prompt = self.embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        # 在輸入序列前添加提示令牌
        x = torch.cat([prompt, x], dim=1)

        # 構建提示令牌的掩碼（提示令牌不需要填充）
        prompt_mask = torch.zeros(
            batch_size, self.n_prompt, dtype=torch.bool, device=x.device
        )
        # 構建原始序列的填充掩碼
        frame_mask = build_padding_mask(
            batch_size, x.size(1) - self.n_prompt, lengths, x.device
        )
        # 合併提示令牌掩碼和原始序列掩碼
        padding_mask = torch.cat([prompt_mask, frame_mask], dim=1)
        return x, padding_mask, self.n_prompt


def build_padding_mask(batch_size, max_len, lengths, device):
    """構建填充掩碼

    Args:
        batch_size: 批次大小
        max_len: 序列最大長度
        lengths: 每個樣本的有效序列長度，形狀為 (batch_size,)
        device: 計算設備

    Returns:
        填充掩碼，形狀為 (batch_size, max_len)，值為True的位置表示填充
    """
    steps = torch.arange(max_len, device=device).unsqueeze(0)
    lengths = lengths.unsqueeze(1)
    return steps >= lengths
