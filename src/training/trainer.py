import json
from pathlib import Path

import torch
from torch import nn

from .metrics import compute_metrics


def move_batch_to_device(batch, device):
    return {
        "features": batch["features"].to(device),
        "lengths": batch["lengths"].to(device),
        "labels": batch["labels"].to(device),
        "domains": batch["domains"].to(device),
        "paths": batch["paths"],
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    num_classes,
    logger,
    log_every=20,
    max_grad_norm=None,
    domain_classifier=None,
    domain_lambda=0.0,
    domain_grl=1.0,
    domain_criterion=None,
):
    model.train()
    if domain_classifier:
        domain_classifier.train()

    total_loss = 0.0
    total_domain_loss = 0.0
    all_preds = []
    all_labels = []

    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        logits, features = model(batch["features"], batch["lengths"])
        loss = criterion(logits, batch["labels"])

        domain_loss = torch.tensor(0.0, device=device)
        if domain_classifier is not None and domain_criterion is not None:
            domain_logits = domain_classifier(features, grl_lambda=domain_grl)
            domain_loss = domain_criterion(domain_logits, batch["domains"])
            loss = loss + domain_lambda * domain_loss

        loss.backward()
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        total_domain_loss += domain_loss.item()

        preds = logits.argmax(dim=1).detach().cpu().tolist()
        labels = batch["labels"].detach().cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels)

        if logger and step % log_every == 0:
            logger.info(
                "step=%s loss=%.4f domain_loss=%.4f",
                step,
                total_loss / step,
                total_domain_loss / step,
            )

    metrics = compute_metrics(all_preds, all_labels, num_classes)
    metrics["loss"] = total_loss / max(1, len(loader))
    if domain_classifier is not None:
        metrics["domain_loss"] = total_domain_loss / max(1, len(loader))
    return metrics


def evaluate(model, loader, criterion, device, num_classes, domain_classifier=None, domain_criterion=None):
    model.eval()
    if domain_classifier:
        domain_classifier.eval()

    total_loss = 0.0
    total_domain_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            logits, features = model(batch["features"], batch["lengths"])
            loss = criterion(logits, batch["labels"])

            domain_loss = torch.tensor(0.0, device=device)
            if domain_classifier is not None and domain_criterion is not None:
                domain_logits = domain_classifier(features, grl_lambda=1.0)
                domain_loss = domain_criterion(domain_logits, batch["domains"])

            total_loss += loss.item()
            total_domain_loss += domain_loss.item()

            preds = logits.argmax(dim=1).cpu().tolist()
            labels = batch["labels"].cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)

    metrics = compute_metrics(all_preds, all_labels, num_classes)
    metrics["loss"] = total_loss / max(1, len(loader))
    if domain_classifier is not None:
        metrics["domain_loss"] = total_domain_loss / max(1, len(loader))
    return metrics


def save_checkpoint(path, model, optimizer, epoch, metrics, label_list, config, domain_classifier=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "labels": label_list,
        "config": config,
    }
    if domain_classifier is not None:
        payload["domain_state"] = domain_classifier.state_dict()
    torch.save(payload, path)


def load_checkpoint(path, model, device, domain_classifier=None):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if domain_classifier is not None and "domain_state" in checkpoint:
        domain_classifier.load_state_dict(checkpoint["domain_state"], strict=True)
    return checkpoint


def save_metrics(path, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
