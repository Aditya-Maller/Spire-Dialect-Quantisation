"""
Backend/quantization.py
=======================
Implements acoustic signal quantization schemes for audio visualization and playback:
- Scheme 1: 1-Bit Sign Binary (+1 / -1 with silence noise gate)
- Scheme 2: Uniform Mid-Tread Linear PCM (2, 4, 8-bit)
- Scheme 3: Mu-law Companding (G.711, mu=255) + Expansion (2, 4, 8-bit)
- Scheme 4: A-law Companding (G.711, A=87.6) + Expansion (2, 4, 8-bit)
- Scheme 5: Logarithmic Base-2 Quantization (2, 4, 8-bit)
- Scheme 6: 16-Bit Full Precision Reference
"""

import io
import numpy as np
import soundfile as sf

MU = 255.0
A = 87.6

def normalise_waveform(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Peak-normalises a 1D audio waveform array to [-1.0, 1.0] and returns peak amplitude."""
    x = np.asarray(x, dtype=np.float64)
    peak = float(np.max(np.abs(x)))
    if peak == 0.0:
        return x, 1.0
    return x / peak, peak

def mu_law_compress(x: np.ndarray, mu: float = MU) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)

def mu_law_expand(y: np.ndarray, mu: float = MU) -> np.ndarray:
    y = np.clip(y, -1.0, 1.0)
    return np.sign(y) * (np.expm1(np.abs(y) * np.log1p(mu))) / mu

def a_law_compress(x: np.ndarray, a: float = A) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    ax = np.abs(x)
    lna = np.log(a)
    out = np.where(
        ax <= 1.0 / a,
        a * ax / (1.0 + lna),
        (1.0 + np.log(a * ax + 1e-12)) / (1.0 + lna),
    )
    return np.sign(x) * np.clip(out, 0.0, 1.0)

def a_law_expand(y: np.ndarray, a: float = A) -> np.ndarray:
    y = np.clip(y, -1.0, 1.0)
    ay = np.abs(y)
    lna = np.log(a)
    threshold = 1.0 / (1.0 + lna)
    out = np.where(
        ay < threshold,
        ay * (1.0 + lna) / a,
        np.exp(ay * (1.0 + lna) - 1.0) / a,
    )
    return np.sign(y) * out

def _uniform_midtread(x: np.ndarray, bit_depth: int) -> np.ndarray:
    delta = 2.0 ** (1 - bit_depth)
    x_q = delta * np.round(x / delta)
    limit = 1.0 - delta / 2.0
    return np.clip(x_q, -limit, limit)

def _log2_quantize(x: np.ndarray, bit_depth: int) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    sign = np.sign(x)
    mag = np.abs(x)

    eps = 2.0 ** -(2 ** (bit_depth - 1))
    mag_safe = np.where(mag < eps, eps, mag)

    log_mag = np.log2(mag_safe)
    log_min = -float(2 ** (bit_depth - 1))
    log_norm = np.clip(log_mag / log_min, 0.0, 1.0)

    levels = 2 ** (bit_depth - 1)
    log_q = np.round(log_norm * (levels - 1)) / (levels - 1)

    mag_q = 2.0 ** (log_q * log_min)
    return np.where(sign == 0, 0.0, sign * mag_q)

def quantize_waveform(
    x: np.ndarray,
    scheme: str,
    bit_depth: int = 1,
    silence_gate: bool = True,
    silence_thresh: float = 0.02
) -> np.ndarray:
    """
    Applies the chosen quantization scheme to 1D audio waveform.
    Returns quantized float64 waveform in [-1.0, 1.0].
    """
    x_norm, _ = normalise_waveform(x)

    if scheme == "1-bit Sign" or bit_depth == 1:
        out = np.where(x_norm >= 0, 1.0, -1.0)
        if silence_gate:
            # Mask out background silence so pauses stay 0.0 instead of loud square wave noise
            out[np.abs(x_norm) < silence_thresh] = 0.0
        return out

    if scheme == "1-bit On-Off Pulse":
        # Threshold On-Off keying (Morse pulse style)
        out = np.where(np.abs(x_norm) > silence_thresh, np.sign(x_norm), 0.0)
        return out

    if scheme == "16-bit Baseline" or bit_depth == 16:
        return x_norm.copy()

    if scheme == "Uniform":
        q = _uniform_midtread(x_norm, bit_depth)
    elif scheme == "Mu-law":
        y_compressed = mu_law_compress(x_norm)
        y_quantized = _uniform_midtread(y_compressed, bit_depth)
        q = mu_law_expand(y_quantized)
    elif scheme == "A-law":
        y_compressed = a_law_compress(x_norm)
        y_quantized = _uniform_midtread(y_compressed, bit_depth)
        q = a_law_expand(y_quantized)
    elif scheme == "Logarithmic":
        q = _log2_quantize(x_norm, bit_depth)
    else:
        q = _uniform_midtread(x_norm, bit_depth)

    if silence_gate and bit_depth <= 2:
        q[np.abs(x_norm) < silence_thresh] = 0.0

    return q

def calculate_snr(original: np.ndarray, quantized: np.ndarray) -> float:
    """Calculates Signal-to-Noise Ratio (SNR) in dB."""
    orig_norm, _ = normalise_waveform(original)
    signal_power = np.mean(orig_norm ** 2)
    noise_power = np.mean((orig_norm - quantized) ** 2)
    if noise_power == 0 or signal_power == 0:
        return 100.0
    snr = 10 * np.log10(signal_power / noise_power)
    return round(float(snr), 2)

def calculate_mse(original: np.ndarray, quantized: np.ndarray) -> float:
    """Calculates Mean Squared Error (MSE)."""
    orig_norm, _ = normalise_waveform(original)
    return round(float(np.mean((orig_norm - quantized) ** 2)), 6)

def waveform_to_wav_bytes(waveform: np.ndarray, sample_rate: int = 16000, orig_peak: float = 0.8) -> bytes:
    """
    Converts float numpy waveform into WAV audio bytes for playback.
    Preserves original speech peak scale so playback volume is natural and crisp.
    """
    buf = io.BytesIO()
    wf = np.clip(waveform, -1.0, 1.0)
    scale = min(max(orig_peak, 0.3), 0.85)
    wf_scaled = wf * scale
    sf.write(buf, wf_scaled.astype(np.float32), sample_rate, format='WAV', subtype='FLOAT')
    return buf.getvalue()
