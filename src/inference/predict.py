"""推理預測模塊

提供模型加載、音頻特征提取和情感預測功能
用於部署和使用訓練好的情感分類模型
"""
from pathlib import Path

import torch
import torchaudio

from src.models.model import EmotionModel


def load_model(checkpoint_path, device):
    """加載訓練好的模型

    Args:
        checkpoint_path: 模型檢查點路徑
        device: 運算設備 ("cpu" 或 "cuda")

    Returns:
        model: 加載好的情感分類模型
        labels: 類別標籤列表
        config: 模型配置字典
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    labels = checkpoint["labels"]

    data_cfg = config["data"]
    model_cfg = config["model"]

    # 創建模型實例
    model = EmotionModel(
        n_mels=data_cfg["n_mels"],
        num_classes=len(labels),
        **model_cfg,
    ).to(device)
    # 加載模型權重
    model.load_state_dict(checkpoint["model_state"], strict=True)
    # 設置模型為評估模式
    model.eval()
    return model, labels, config


def extract_features(path, sample_rate, n_mels, max_duration=None, n_fft=None, hop_length=None):
    """從音頻文件中提取特征

    Args:
        path: 音頻文件路徑
        sample_rate: 目標采樣率
        n_mels: Mel頻譜的維度
        max_duration: 最大音頻持續時間（秒），超過則裁剪，不足則填充
        n_fft: FFT窗口大小
        hop_length: 跳躍長度

    Returns:
        features: 提取的Mel頻譜特征，形狀為 (1, seq_length, n_mels)
        lengths: 特征序列長度，形狀為 (1,)
    """
    # 加載音頻文件
    waveform, sr = torchaudio.load(path)
    # 轉換為單通道
    waveform = waveform.mean(dim=0, keepdim=True)
    # 重采樣到目標采樣率
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    # 裁剪或填充到指定長度
    if max_duration:
        max_samples = int(max_duration * sample_rate)
        if waveform.shape[-1] > max_samples:
            waveform = waveform[:, :max_samples]
        elif waveform.shape[-1] < max_samples:
            pad = max_samples - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

    # 配置Mel頻譜轉換器
    mel_kwargs = {"sample_rate": sample_rate, "n_mels": n_mels}
    if n_fft:
        mel_kwargs["n_fft"] = n_fft
    if hop_length:
        mel_kwargs["hop_length"] = hop_length
    mel = torchaudio.transforms.MelSpectrogram(**mel_kwargs)
    # 轉換為dB刻度
    to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    features = to_db(mel(waveform)).squeeze(0).transpose(0, 1)
    lengths = torch.tensor([features.shape[0]], dtype=torch.long)
    return features.unsqueeze(0), lengths


def predict_file(checkpoint_path, audio_path, device="cpu"):
    """預測音頻文件的情感

    Args:
        checkpoint_path: 模型檢查點路徑
        audio_path: 音頻文件路徑
        device: 運算設備 ("cpu" 或 "cuda")

    Returns:
        情感類別和對應概率的字典
    """
    audio_path = Path(audio_path)
    # 加載模型
    model, labels, config = load_model(checkpoint_path, device)
    data_cfg = config["data"]

    # 提取特征
    features, lengths = extract_features(
        audio_path,
        sample_rate=data_cfg["sample_rate"],
        n_mels=data_cfg["n_mels"],
        max_duration=data_cfg.get("max_duration"),
        n_fft=data_cfg.get("n_fft"),
        hop_length=data_cfg.get("hop_length"),
    )
    # 移動到指定設備
    features = features.to(device)
    lengths = lengths.to(device)

    # 模型預測
    with torch.no_grad():
        logits, _ = model(features, lengths)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    # 格式化解果
    return {label: float(prob) for label, prob in zip(labels, probs)}
