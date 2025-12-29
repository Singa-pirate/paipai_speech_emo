"""損失函數模塊

提供損失函數構建和類別權重計算功能
用於處理不平衡數據集和改進模型訓練
"""
import torch


def compute_class_weights(label_ids, num_classes):
    """計算類別權重

    基於數據集中各類別的出現頻率計算權重
    用於解決不平衡數據集的問題

    Args:
        label_ids: 所有樣本的標籤ID列表
        num_classes: 總類別數

    Returns:
        類別權重向量，形狀為 (num_classes,)
    """
    labels = torch.tensor(label_ids, dtype=torch.long)
    # 統計每個類別的出現次數
    counts = torch.bincount(labels, minlength=num_classes).float()
    # 避免除零錯誤，確保每個類別至少有1次出現
    counts = counts.clamp(min=1.0)
    # 計算權重（總樣本數 / 每個類別的樣本數）
    weights = counts.sum() / counts
    # 正規化權重，使其均值為1
    weights = weights / weights.mean()
    return weights


def build_criterion(label_smoothing=0.0, class_weights=None):
    """構建分類損失函數

    Args:
        label_smoothing: 標籤平滑係數，用於正則化
        class_weights: 類別權重向量

    Returns:
        配置好的CrossEntropyLoss損失函數
    """
    if class_weights is not None:
        class_weights = class_weights.float()
    # 創建帶有權重和標籤平滑的交叉熵損失函數
    return torch.nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=label_smoothing
    )
