"""評估指標模塊

提供多種分類任務的評估指標計算功能
包括準確率、宏F1分數、未加權準確率等
"""
import torch


def confusion_matrix(preds, labels, num_classes):
    """計算混淆矩陣

    Args:
        preds: 預測標籤列表
        labels: 真實標籤列表
        num_classes: 總類別數

    Returns:
        混淆矩陣，形狀為 (num_classes, num_classes)
        matrix[i][j] 表示真實標籤為i且預測為j的樣本數
    """
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for label, pred in zip(labels, preds):
        matrix[label, pred] += 1
    return matrix


def accuracy(preds, labels):
    """計算分類準確率

    Args:
        preds: 預測標籤列表
        labels: 真實標籤列表

    Returns:
        準確率，值在0~1之間
    """
    return (preds == labels).float().mean().item()


def macro_f1(preds, labels, num_classes):
    """計算宏F1分數

    宏F1是各類別F1分數的算術平均值
    用於評估模型在不平衡數據集上的整體性能

    Args:
        preds: 預測標籤列表
        labels: 真實標籤列表
        num_classes: 總類別數

    Returns:
        宏F1分數，值在0~1之間
    """
    cm = confusion_matrix(preds, labels, num_classes).float()
    tp = cm.diag()  # 真正例
    precision = tp / (cm.sum(dim=0).clamp(min=1.0))  # 精確率
    recall = tp / (cm.sum(dim=1).clamp(min=1.0))  # 召回率
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-6)  # F1分數
    return f1.mean().item()


def unweighted_accuracy(preds, labels, num_classes):
    """計算未加權準確率（平均召回率）

    未加權準確率是各類別召回率的平均值
    用於評估模型對每個類別的識別能力，不受類別不平衡影響

    Args:
        preds: 預測標籤列表
        labels: 真實標籤列表
        num_classes: 總類別數

    Returns:
        未加權準確率，值在0~1之間
    """
    cm = confusion_matrix(preds, labels, num_classes).float()
    recall = cm.diag() / cm.sum(dim=1).clamp(min=1.0)  # 各類別召回率
    return recall.mean().item()


def compute_metrics(preds, labels, num_classes):
    """計算所有評估指標

    Args:
        preds: 預測標籤列表
        labels: 真實標籤列表
        num_classes: 總類別數

    Returns:
        包含所有指標的字典
    """
    preds = torch.tensor(preds, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    return {
        "accuracy": accuracy(preds, labels),
        "macro_f1": macro_f1(preds, labels, num_classes),
        "ua": unweighted_accuracy(preds, labels, num_classes),
    }
