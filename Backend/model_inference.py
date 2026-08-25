import os
import time
import numpy as np
import soundfile as sf
import io

# We will dynamically import torch/torchaudio/onnxruntime inside helper functions or at top
try:
    import torch
    import torchaudio
except ImportError:
    torch = None
    torchaudio = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

DIALECT_INFO = {
    0: {
        "id": 0,
        "name": "South Karnataka (Mysuru - Bengaluru)",
        "kannada": "ದಕ್ಷಿಣ ಕರ್ನಾಟಕ (ಮೈಸೂರು - ಬೆಂಗಳೂರು)",
        "icon": "🌆",
        "color": "#3B82F6",
        "gradient": "linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%)",
        "description": "Standard southern dialect spoken across Mysuru, Mandya, and Bengaluru urban regions. Marked by polite honorific suffixes and classic vowel length."
    },
    1: {
        "id": 1,
        "name": "Coastal Karnataka (Mangaluru / Karavali)",
        "kannada": "ಕರಾವಳಿ ಕರ್ನಾಟಕ (ಮಂಗಳೂರು / ಉಡುಪಿ)",
        "icon": "🌊",
        "color": "#06B6D4",
        "gradient": "linear-gradient(135deg, #0E7490 0%, #06B6D4 100%)",
        "description": "Coastal Karavali dialect spoken in Mangaluru and Udupi, rich in melodic intonation with phonological influence from Tulu and Konkani."
    },
    2: {
        "id": 2,
        "name": "North-West Karnataka (Hubballi - Dharwad / Belagavi)",
        "kannada": "ಉತ್ತರ-ಪಶ್ಚಿಮ ಕರ್ನಾಟಕ (ಹುಬ್ಬಳ್ಳಿ - ಧಾರವಾಡ)",
        "icon": "🏰",
        "color": "#10B981",
        "gradient": "linear-gradient(135deg, #047857 0%, #10B981 100%)",
        "description": "North-Western region dialect of Dharwad and Belagavi, featuring fast-paced rhythmic cadence and historical linguistic contact features."
    },
    3: {
        "id": 3,
        "name": "North-East Karnataka (Kalaburagi / Hyderabad-Karnataka)",
        "kannada": "ಉತ್ತರ-ಪೂರ್ವ ಕರ್ನಾಟಕ (ಕಲಬುರಗಿ)",
        "icon": "☀️",
        "color": "#F59E0B",
        "gradient": "linear-gradient(135deg, #B45309 0%, #F59E0B 100%)",
        "description": "Kalyana Karnataka dialect spoken in Kalaburagi, Yadgir, and Bidar, characterized by distinct vocabulary and regional cadence."
    },
    4: {
        "id": 4,
        "name": "Central Karnataka (Shivamogga - Chitradurga)",
        "kannada": "ಮಧ್ಯ ಕರ್ನಾಟಕ (ಶಿವಮೊಗ್ಗ - ಚಿತ್ರದುರ್ಗ)",
        "icon": "⛰️",
        "color": "#8B5CF6",
        "gradient": "linear-gradient(135deg, #6D28D9 0%, #8B5CF6 100%)",
        "description": "Central belt dialect bridging the northern and southern phonetic features across Shivamogga, Davanagere, and Chitradurga."
    }
}

MODELS_DIR = os.path.join(os.path.dirname(__file__), "Tlearn_Models")

def get_available_models():
    """Finds all ONNX model files in the Backend/Tlearn_Models directory."""
    if not os.path.exists(MODELS_DIR):
        return {}
    
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".onnx")]
    models = {}
    for f in sorted(files):
        path = os.path.join(MODELS_DIR, f)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        
        # Categorize quantization
        if "16bit" in f or "baseline" in f:
            q_type = "Full Precision (FP16/FP32 Baseline)"
            badge = "16-BIT BASELINE"
        elif "8bit" in f:
            q_type = "8-bit Quantized"
            badge = "8-BIT QUANT"
        elif "4bit" in f:
            q_type = "4-bit Quantized"
            badge = "4-BIT QUANT"
        elif "2bit" in f:
            q_type = "2-bit Quantized"
            badge = "2-BIT QUANT"
        elif "1bit" in f:
            q_type = "1-bit Sign Binary"
            badge = "1-BIT BINARY"
        else:
            q_type = "Quantized Model"
            badge = "ONNX MODEL"
            
        models[f] = {
            "name": f,
            "path": path,
            "size_mb": round(size_mb, 2),
            "type": q_type,
            "badge": badge
        }
    return models

def preprocess_audio_bytes(audio_bytes: bytes, target_sr: int = 16000):
    """
    Decodes audio bytes (wav/mp3/flac) into mono float32 waveform tensor at 16kHz.
    """
    buf = io.BytesIO(audio_bytes)
    data, sr = sf.read(buf, dtype='float32')
    
    if data.ndim > 1:
        data = np.mean(data, axis=1)
        
    waveform = torch.from_numpy(data).unsqueeze(0) # (1, T)
    
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        
    return waveform

def compute_mel_spectrogram(waveform: torch.Tensor) -> torch.Tensor:
    """
    Computes NeMo Conformer compatible Log-Mel Spectrogram.
    Input: waveform shape (1, T) at 16000Hz
    Output: normalized_mel shape (1, 80, time_steps)
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # 1. Pre-emphasis filter: y[t] = x[t] - 0.97 * x[t-1]
    padded = torch.cat([waveform[:, :1], waveform], dim=-1)
    preemphasized = padded[:, 1:] - 0.97 * padded[:, :-1]

    # 2. Mel transform (STFT with n_fft=512, win_length=400, hop_length=160, n_mels=80)
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

    # 3. Log transform
    log_mel = torch.log(mel_spec + 1e-5)

    # 4. Normalization along time axis
    mean = log_mel.mean(dim=-1, keepdim=True)
    std = log_mel.std(dim=-1, keepdim=True) + 1e-5
    normalized_mel = (log_mel - mean) / std

    return normalized_mel

def run_dialect_inference(model_path: str, audio_bytes: bytes):
    """
    Runs full inference pipeline: Audio Bytes -> Waveform -> Mel Spectrogram -> ONNX Model -> Probabilities
    """
    t_start = time.time()
    
    # 1. Preprocessing
    t_prep_0 = time.time()
    waveform = preprocess_audio_bytes(audio_bytes)
    mel_spec = compute_mel_spectrogram(waveform)
    t_prep_ms = (time.time() - t_prep_0) * 1000.0
    
    # Prepare ONNX numpy inputs
    mel_np = mel_spec.numpy() # (1, 80, time_steps)
    seq_len_np = np.array([mel_np.shape[2]], dtype=np.int64)
    
    # 2. ONNX Inference Session Execution
    t_infer_0 = time.time()
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    ort_inputs = {
        "mel_spectrogram": mel_np,
        "sequence_length": seq_len_np
    }
    
    logits = session.run(["dialect_logits"], ort_inputs)[0] # (1, 5)
    t_infer_ms = (time.time() - t_infer_0) * 1000.0
    
    # 3. Postprocessing & Softmax
    raw_logits = logits[0]
    # Softmax
    exp_logits = np.exp(raw_logits - np.max(raw_logits))
    probs = exp_logits / np.sum(exp_logits)
    
    predicted_id = int(np.argmax(probs))
    confidence = float(probs[predicted_id] * 100.0)
    
    total_time_ms = (time.time() - t_start) * 1000.0
    
    # Format probabilities distribution
    distribution = []
    for i in range(len(probs)):
        info = DIALECT_INFO.get(i, {})
        distribution.append({
            "id": i,
            "name": info.get("name", f"Dialect {i}"),
            "kannada": info.get("kannada", ""),
            "prob": float(probs[i]),
            "percent": round(float(probs[i] * 100.0), 2),
            "color": info.get("color", "#3B82F6"),
            "icon": info.get("icon", "🎙️")
        })
        
    # Sort distribution by confidence descending
    distribution = sorted(distribution, key=lambda x: x["prob"], reverse=True)
    
    audio_duration_sec = waveform.shape[1] / 16000.0
    
    return {
        "predicted_id": predicted_id,
        "predicted_dialect": DIALECT_INFO[predicted_id],
        "confidence": round(confidence, 2),
        "distribution": distribution,
        "audio_duration_sec": round(audio_duration_sec, 2),
        "latency": {
            "preprocess_ms": round(t_prep_ms, 2),
            "inference_ms": round(t_infer_ms, 2),
            "total_ms": round(total_time_ms, 2),
            "realtime_factor": round((total_time_ms / 1000.0) / max(audio_duration_sec, 0.001), 3)
        },
        "waveform_sample": waveform.squeeze(0).numpy()[::max(1, int(waveform.shape[1]/500))] # Downsampled for UI plot
    }
