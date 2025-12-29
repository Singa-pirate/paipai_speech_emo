"""訓練器模塊

提供模型訓練、評估、檢查點保存和加載功能
用於管理完整的模型訓練流程
"""
import json
from pathlib import Path

import torch
from torch import nn

from .metrics import compute_metrics


def move_batch_to_device(batch, device):
    """將批次數據移動到指定設備

    Args:
        batch: 批次數據字典
        device: 目標設備（"cpu" 或 "cuda"）

    Returns:
        移動到指定設備後的批次數據字典
    """
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
    """訓練模型一個epoch

    Args:
        model: 情感分類模型
        loader: 訓練數據加載器
        optimizer: 優化器
        criterion: 主損失函數
        device: 運算設備
        num_classes: 類別數量
        logger: 日誌記錄器
        log_every: 日誌記錄間隔
        max_grad_norm: 梯度裁剪閾值
        domain_classifier: 領域分類器（可選）
        domain_lambda: 領域損失權重
        domain_grl: 梯度反轉層係數
        domain_criterion: 領域分類損失函數

    Returns:
        訓練指標字典
    """
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
    """評估模型性能

    Args:
        model: 情感分類模型
        loader: 測試數據加載器
        criterion: 主損失函數
        device: 運算設備
        num_classes: 類別數量
        domain_classifier: 領域分類器（可選）
        domain_criterion: 領域分類損失函數

    Returns:
        評估指標字典
    """
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
    """保存模型檢查點

    Args:
        path: 檢查點保存路徑
        model: 情感分類模型
        optimizer: 優化器
        epoch: 當前epoch數
        metrics: 訓練指標
        label_list: 類別標籤列表
        config: 模型配置
        domain_classifier: 領域分類器（可選）
    """
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
    """加載模型檢查點

    Args:
        path: 檢查點路徑
        model: 情感分類模型
        device: 運算設備
        domain_classifier: 領域分類器（可選）

    Returns:
        檢查點數據字典
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if domain_classifier is not None and "domain_state" in checkpoint:
        domain_classifier.load_state_dict(checkpoint["domain_state"], strict=True)
    return checkpoint


def save_metrics(path, metrics):
    """保存評估指標到JSON文件

    Args:
        path: 保存路徑
        metrics: 指標字典
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
