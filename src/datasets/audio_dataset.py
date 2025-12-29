import csv
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset

from .augment import apply_augmentations
from .label_mapping import load_label_map, map_label

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a"}


def _resolve_audio_path(path_value, root_dir=None):
    if path_value is None:
        raise ValueError("Missing audio path in manifest")
    path = Path(path_value)
    if not path.is_absolute() and root_dir:
        path = Path(root_dir) / path
    return path


def _read_manifest(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Manifest not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest is empty: {csv_path}")
    return rows


def _get_field(row, candidates):
    for key in candidates:
        if key in row and row[key] != "":
            return row[key]
    return None


class EmotionDataset(Dataset):
    def __init__(
        self,
        csv_path,
        label_list,
        label_map_path=None,
        sample_rate=16000,
        n_mels=80,
        n_fft=None,
        hop_length=None,
        max_duration=6.0,
        min_duration=0.5,
        mode="train",
        augment=None,
        root_dir=None,
        domain_id=None,
        drop_unknown=True,
    ):
        self.csv_path = Path(csv_path)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_samples = int(max_duration * sample_rate) if max_duration else None
        self.min_samples = int(min_duration * sample_rate) if min_duration else None
        self.mode = mode
        self.augment = augment if mode == "train" else None
        self.root_dir = root_dir
        self.drop_unknown = drop_unknown
        self.label_list = list(label_list)
        self.label_to_id = {label: idx for idx, label in enumerate(self.label_list)}

        self.label_map = load_label_map(label_map_path) if label_map_path else {}

        self.items = []
        rows = _read_manifest(self.csv_path)
        for row in rows:
            path_value = _get_field(row, ["path", "filepath", "audio", "wav", "file"])
            label_value = _get_field(row, ["label", "emotion", "y"])
            speaker = _get_field(row, ["speaker", "spk", "speaker_id"]) or ""
            domain = _get_field(row, ["domain", "domain_id"]) or domain_id

            if label_value is None or path_value is None:
                continue

            mapped_label = map_label(label_value, self.label_map)
            if mapped_label not in self.label_to_id:
                if drop_unknown:
                    continue
                raise ValueError(f"Unknown label '{mapped_label}' in {self.csv_path}")

            resolved = _resolve_audio_path(path_value, root_dir)
            if resolved.suffix.lower() not in AUDIO_EXTS:
                continue

            self.items.append(
                {
                    "path": str(resolved),
                    "label": self.label_to_id[mapped_label],
                    "speaker": speaker,
                    "domain": int(domain) if domain is not None else 0,
                }
            )

        if not self.items:
            raise ValueError(f"No usable samples found in {self.csv_path}")

        mel_kwargs = {"sample_rate": sample_rate, "n_mels": n_mels}
        if n_fft:
            mel_kwargs["n_fft"] = n_fft
        if hop_length:
            mel_kwargs["hop_length"] = hop_length
        self.mel = torchaudio.transforms.MelSpectrogram(**mel_kwargs)
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        waveform, sr = torchaudio.load(item["path"])
        waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.sample_rate
            )

        if self.augment:
            waveform = apply_augmentations(waveform, self.sample_rate, self.augment)

        waveform = self._trim_or_pad(waveform)

        features = self.to_db(self.mel(waveform))
        features = features.squeeze(0).transpose(0, 1)
        length = features.shape[0]

        return {
            "features": features,
            "length": length,
            "label": item["label"],
            "domain": item["domain"],
            "path": item["path"],
        }

    def _trim_or_pad(self, waveform):
        num_samples = waveform.shape[-1]
        if self.min_samples and num_samples < self.min_samples:
            pad_amount = self.min_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            num_samples = waveform.shape[-1]

        if self.max_samples and num_samples > self.max_samples:
            if self.mode == "train":
                start = torch.randint(0, num_samples - self.max_samples + 1, (1,)).item()
            else:
                start = max(0, (num_samples - self.max_samples) // 2)
            waveform = waveform[:, start : start + self.max_samples]
            num_samples = waveform.shape[-1]

        if self.max_samples and num_samples < self.max_samples:
            pad_amount = self.max_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        return waveform

    def get_label_ids(self):
        return [item["label"] for item in self.items]


def collate_batch(batch):
    max_len = max(sample["length"] for sample in batch)
    feat_dim = batch[0]["features"].shape[-1]

    features = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    labels = torch.zeros(len(batch), dtype=torch.long)
    domains = torch.zeros(len(batch), dtype=torch.long)
    paths = []

    for idx, sample in enumerate(batch):
        length = sample["length"]
        features[idx, :length] = sample["features"]
        lengths[idx] = length
        labels[idx] = sample["label"]
        domains[idx] = sample["domain"]
        paths.append(sample["path"])

    return {
        "features": features,
        "lengths": lengths,
        "labels": labels,
        "domains": domains,
        "paths": paths,
    }
