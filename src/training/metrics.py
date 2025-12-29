import torch


def confusion_matrix(preds, labels, num_classes):
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for label, pred in zip(labels, preds):
        matrix[label, pred] += 1
    return matrix


def accuracy(preds, labels):
    return (preds == labels).float().mean().item()


def macro_f1(preds, labels, num_classes):
    cm = confusion_matrix(preds, labels, num_classes).float()
    tp = cm.diag()
    precision = tp / (cm.sum(dim=0).clamp(min=1.0))
    recall = tp / (cm.sum(dim=1).clamp(min=1.0))
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-6)
    return f1.mean().item()


def unweighted_accuracy(preds, labels, num_classes):
    cm = confusion_matrix(preds, labels, num_classes).float()
    recall = cm.diag() / cm.sum(dim=1).clamp(min=1.0)
    return recall.mean().item()


def compute_metrics(preds, labels, num_classes):
    preds = torch.tensor(preds, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    return {
        "accuracy": accuracy(preds, labels),
        "macro_f1": macro_f1(preds, labels, num_classes),
        "ua": unweighted_accuracy(preds, labels, num_classes),
    }
