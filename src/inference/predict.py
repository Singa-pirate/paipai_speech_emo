from pathlib import Path

import torch
import torchaudio

from src.models.model import EmotionModel


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    labels = checkpoint["labels"]

    data_cfg = config["data"]
    model_cfg = config["model"]

    model = EmotionModel(
        n_mels=data_cfg["n_mels"],
        num_classes=len(labels),
        **model_cfg,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, labels, config


def extract_features(path, sample_rate, n_mels, max_duration=None, n_fft=None, hop_length=None):
    waveform, sr = torchaudio.load(path)
    waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    if max_duration:
        max_samples = int(max_duration * sample_rate)
        if waveform.shape[-1] > max_samples:
            waveform = waveform[:, :max_samples]
        elif waveform.shape[-1] < max_samples:
            pad = max_samples - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

    mel_kwargs = {"sample_rate": sample_rate, "n_mels": n_mels}
    if n_fft:
        mel_kwargs["n_fft"] = n_fft
    if hop_length:
        mel_kwargs["hop_length"] = hop_length
    mel = torchaudio.transforms.MelSpectrogram(**mel_kwargs)
    to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    features = to_db(mel(waveform)).squeeze(0).transpose(0, 1)
    lengths = torch.tensor([features.shape[0]], dtype=torch.long)
    return features.unsqueeze(0), lengths


def predict_file(checkpoint_path, audio_path, device="cpu"):
    audio_path = Path(audio_path)
    model, labels, config = load_model(checkpoint_path, device)
    data_cfg = config["data"]

    features, lengths = extract_features(
        audio_path,
        sample_rate=data_cfg["sample_rate"],
        n_mels=data_cfg["n_mels"],
        max_duration=data_cfg.get("max_duration"),
        n_fft=data_cfg.get("n_fft"),
        hop_length=data_cfg.get("hop_length"),
    )
    features = features.to(device)
    lengths = lengths.to(device)

    with torch.no_grad():
        logits, _ = model(features, lengths)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    return {label: float(prob) for label, prob in zip(labels, probs)}
