"""情感分類模型模塊

實現完整的情感分類模型，集成Transformer骨架、提示令牌和分類頭
用於從音頻特征預測情感標籤
"""
from torch import nn

from .backbone import TransformerBackbone, build_sinusoidal_positions
from .head import ClassificationHead
from .prompt import PromptTokens


def mean_pool(features, padding_mask):
    """對特征序列進行均值池化

    Args:
        features: 輸入特征序列，形狀為 (batch_size, seq_length, feature_dim)
        padding_mask: 填充掩碼，形狀為 (batch_size, seq_length)，值為True的位置表示填充

    Returns:
        池化後的特征，形狀為 (batch_size, feature_dim)
    """
    if padding_mask is None:
        return features.mean(dim=1)
    valid = (~padding_mask).float().unsqueeze(-1)
    summed = (features * valid).sum(dim=1)
    denom = valid.sum(dim=1).clamp(min=1.0)
    return summed / denom


class EmotionModel(nn.Module):
    """情感分類模型

    集成Transformer骨架、提示令牌和分類頭的完整情感分類模型

    Args:
        n_mels: Mel頻譜特征的維度
        num_classes: 情感類別數
        d_model: 模型嵌入維度
        n_heads: 注意力頭數量
        n_layers: Transformer層數
        ff_mult: 前饋網絡隱藏層擴展倍數
        dropout: Dropout概率
        n_prompt: 提示令牌的數量
        pool: 池化方式，"mean"或"cls"
    """
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
        # 初始化提示令牌
        self.prompt = PromptTokens(n_prompt, d_model)
        # 初始化Transformer骨架
        self.backbone = TransformerBackbone(
            input_dim=n_mels,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_mult=ff_mult,
            dropout=dropout,
        )
        # 初始化分類頭
        self.head = ClassificationHead(d_model, num_classes, dropout=dropout)
        self.embedding_dim = d_model

    def forward(self, features, lengths):
        """前向傳播

        Args:
            features: 輸入特征，形狀為 (batch_size, seq_length, n_mels)
            lengths: 每個樣本的有效序列長度，形狀為 (batch_size,)

        Returns:
            logits: 類別分數，形狀為 (batch_size, num_classes)
            pooled: 池化後的特征，形狀為 (batch_size, d_model)
        """
        # 輸入投影
        x = self.backbone.input_proj(features)
        # 添加位置編碼
        pos = build_sinusoidal_positions(x.size(1), x.size(2), x.device)
        x = x + pos.unsqueeze(0)
        # 添加提示令牌
        x, padding_mask, prompt_len = self.prompt(x, lengths)
        # 通過Transformer編碼器
        x = self.backbone.encoder(
            self.backbone.dropout(x),
            src_key_padding_mask=padding_mask,
        )

        # 提取真實音頻特征部分（去除提示令牌）
        if prompt_len > 0:
            frame_x = x[:, prompt_len:, :]
            frame_mask = padding_mask[:, prompt_len:]
        else:
            frame_x = x
            frame_mask = padding_mask

        # 特征池化
        if self.pool == "cls" and prompt_len > 0:
            pooled = x[:, 0, :]
        else:
            pooled = mean_pool(frame_x, frame_mask)
        # 分類預測
        logits = self.head(pooled)
        return logits, pooled

    def get_param_groups(self):
        """獲取參數組

        用於優化器的參數分組，方便進行不同的學習率設置

        Returns:
            encoder_params: 骨架網絡的參數列表
            head_params: 分類頭和提示令牌的參數列表
        """
        encoder_params = list(self.backbone.parameters())
        head_params = list(self.head.parameters())
        if self.prompt.embeddings is not None:
            head_params.append(self.prompt.embeddings)
        return encoder_params, head_params
