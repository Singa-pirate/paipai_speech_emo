"""模型評估腳本

用於評估訓練好的情感分類模型在測試集上的性能
支持加載檢查點、領域適應模型和多種評估指標
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch

from src.datasets.audio_dataset import EmotionDataset, collate_batch
from src.training.loss import build_criterion
from src.training.trainer import evaluate, load_checkpoint
from src.utils.config import load_config
from src.models.model import EmotionModel
from src.models.domain import DomainClassifier


def main():
    """主函數

    解析命令行參數，加載配置和模型，評估模型性能

    命令行參數：
        --config: 配置文件路徑
        --checkpoint: 模型檢查點路徑
        --split: 評估數據集，可選值為"val"或"test"，默認為"test"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="配置文件路徑")
    parser.add_argument("--checkpoint", required=True, help="模型檢查點路徑")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="評估數據集")
    args = parser.parse_args()

    # 加載配置文件
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = ROOT / args.config
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 準備數據集
    data_cfg = cfg["data"]
    csv_path = data_cfg.get(f"{args.split}_csv")
    if not csv_path:
        raise ValueError(f"{args.split}_csv not found in config")

    label_map = data_cfg.get(f"{args.split}_label_map") or data_cfg.get("label_map")

    dataset = EmotionDataset(
        csv_path=csv_path,
        label_list=cfg["labels"],
        label_map_path=label_map,
        sample_rate=data_cfg["sample_rate"],
        n_mels=data_cfg["n_mels"],
        n_fft=data_cfg.get("n_fft"),
        hop_length=data_cfg.get("hop_length"),
        max_duration=data_cfg.get("max_duration", 6.0),
        min_duration=data_cfg.get("min_duration", 0.5),
        mode=args.split,
        root_dir=data_cfg.get("root_dir"),
        domain_id=data_cfg.get("domain_id", 0),
    )

    # 創建數據加載器
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 2),
        collate_fn=collate_batch,
    )

    # 初始化模型
    model = EmotionModel(
        n_mels=data_cfg["n_mels"],
        num_classes=len(cfg["labels"]),
        **cfg["model"],
    ).to(device)

    # 初始化領域分類器（如果需要）
    domain_classifier = None
    if cfg["training"].get("domain_adaptation", False):
        domain_classifier = DomainClassifier(
            input_dim=model.embedding_dim,
            hidden_dim=cfg["training"].get("domain_hidden", 128),
            num_domains=cfg["training"].get("num_domains", 2),
        ).to(device)

    # 加載檢查點
    load_checkpoint(args.checkpoint, model, device, domain_classifier=domain_classifier)
    
    # 創建損失函數
    criterion = build_criterion(label_smoothing=0.0, class_weights=None)
    
    # 評估模型
    metrics = evaluate(
        model,
        loader,
        criterion,
        device,
        len(cfg["labels"]),
        domain_classifier=domain_classifier,
    )
    
    print(metrics)


if __name__ == "__main__":
    main()
