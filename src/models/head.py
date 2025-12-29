"""分類頭模塊

提供分類任務所需的網絡頭部
將特征映射到分類概率空間
"""
from torch import nn


class ClassificationHead(nn.Module):
    """分類頭

    用於將提取的特征映射到類別概率分布
    包含層歸一化、隱藏層、激活函數和Dropout

    Args:
        input_dim: 輸入特征維度
        num_classes: 分類類別數
        hidden_dim: 隱藏層維度，默認為input_dim
        dropout: Dropout概率
    """
    def __init__(self, input_dim, num_classes, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),  # 層歸一化
            nn.Linear(input_dim, hidden_dim),  # 線性變換到隱藏層
            nn.GELU(),  # GELU激活函數
            nn.Dropout(dropout),  # Dropout正則化
            nn.Linear(hidden_dim, num_classes),  # 輸出層，生成類別分數
        )

    def forward(self, x):
        """前向傳播

        Args:
            x: 輸入特征，形狀為 (batch_size, input_dim)

        Returns:
            類別分數，形狀為 (batch_size, num_classes)
        """
        return self.net(x)
