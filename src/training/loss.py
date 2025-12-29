import torch


def compute_class_weights(label_ids, num_classes):
    labels = torch.tensor(label_ids, dtype=torch.long)
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = counts.clamp(min=1.0)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return weights


def build_criterion(label_smoothing=0.0, class_weights=None):
    if class_weights is not None:
        class_weights = class_weights.float()
    return torch.nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=label_smoothing
    )
