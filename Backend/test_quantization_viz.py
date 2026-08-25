import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import io
import soundfile as sf
from Backend.quantization import quantize_waveform, calculate_snr, calculate_mse, waveform_to_wav_bytes

def test_quant():
    sr = 16000
    t = np.linspace(0, 1.0, sr)
    env = np.exp(-3 * t) * np.sin(2 * np.pi * 3 * t)
    x = env * np.sin(2 * np.pi * 440 * t)
    x = x / np.max(np.abs(x)) # [-1, 1]
    
    print("Testing quantization schemes...")
    for scheme in ["1-bit Sign", "Uniform", "Mu-law", "A-law", "Logarithmic", "16-bit Baseline"]:
        for bd in [1, 2, 4, 8, 16]:
            if scheme == "1-bit Sign" and bd != 1:
                continue
            if scheme == "16-bit Baseline" and bd != 16:
                continue
            if scheme not in ["1-bit Sign", "16-bit Baseline"] and bd in [1, 16]:
                continue
            q = quantize_waveform(x, scheme, bd)
            snr = calculate_snr(x, q)
            mse = calculate_mse(x, q)
            wav_b = waveform_to_wav_bytes(q, sr)
            print(f"Scheme: {scheme:15s} | Bits: {bd:2d} | SNR: {snr:6.2f} dB | MSE: {mse:.6f} | Bytes: {len(wav_b)}")

if __name__ == "__main__":
    test_quant()
