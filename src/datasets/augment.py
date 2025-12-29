import random

import torch
import torchaudio


def apply_noise(waveform, noise_std):
    if noise_std <= 0:
        return waveform
    noise = torch.randn_like(waveform) * noise_std
    return waveform + noise


def apply_time_shift(waveform, max_shift):
    if max_shift <= 0:
        return waveform
    shift = int(random.uniform(-max_shift, max_shift) * waveform.shape[-1])
    if shift == 0:
        return waveform
    return torch.roll(waveform, shifts=shift, dims=-1)


def apply_speed_perturb(waveform, sample_rate, speeds):
    if not speeds:
        return waveform
    speed = random.choice(speeds)
    if speed == 1.0:
        return waveform
    new_rate = int(sample_rate * speed)
    waveform = torchaudio.functional.resample(waveform, sample_rate, new_rate)
    waveform = torchaudio.functional.resample(waveform, new_rate, sample_rate)
    return waveform


def apply_augmentations(waveform, sample_rate, config):
    if not config or not config.get("enabled", False):
        return waveform
    waveform = apply_speed_perturb(
        waveform, sample_rate, config.get("speed_perturb", [])
    )
    waveform = apply_time_shift(waveform, config.get("time_shift", 0.0))
    waveform = apply_noise(waveform, config.get("noise_std", 0.0))
    return waveform
