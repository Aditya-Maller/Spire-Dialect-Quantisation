import os
import torch
import torchaudio
import numpy as np

def compute_nemo_mel_spectrogram(waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    """
    Computes NeMo-compatible Log-Mel-Spectrogram for Conformer models.
    Inputs:
        waveform: PyTorch tensor of shape (1, T) or (T,) at 16kHz
    Outputs:
        mel_spec: PyTorch tensor of shape (1, 80, time_steps)
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 2 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # 1. Pre-emphasis filter: y[t] = x[t] - 0.97 * x[t-1]
    padded_wave = torch.cat([waveform[:, :1], waveform], dim=-1)
    preemphasized = padded_wave[:, 1:] - 0.97 * padded_wave[:, :-1]

    # 2. STFT & Mel Spectrogram
    # n_fft=512, win_length=400 (25ms), hop_length=160 (10ms), n_mels=80
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        f_min=0.0,
        f_max=8000.0,
        n_mels=80,
        window_fn=torch.hann_window,
        power=2.0,
        center=True
    )
    
    mel_spec = mel_transform(preemphasized)
    
    # 3. Log transform log(mel + 1e-5)
    log_mel = torch.log(mel_spec + 1e-5)

    # 4. Per-feature normalization (zero mean, unit variance along time)
    mean = log_mel.mean(dim=-1, keepdim=True)
    std = log_mel.std(dim=-1, keepdim=True) + 1e-5
    normalized_mel = (log_mel - mean) / std

    return normalized_mel

if __name__ == "__main__":
    print("Testing Mel Spectrogram function...")
    dummy_audio = torch.randn(1, 16000 * 3) # 3 seconds audio
    mel = compute_nemo_mel_spectrogram(dummy_audio)
    print(f"Mel shape: {mel.shape}") # Expect (1, 80, time_steps)
