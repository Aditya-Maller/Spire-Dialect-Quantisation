# 🎙️ SPIRE: NeMo Conformer Kannada Dialect AI App

An interactive, high-performance Streamlit presentation application for live speech recognition and Kannada dialect classification using fine-tuned NeMo Conformer ONNX models (IISc RESPIN Corpus).

---

## ✨ Features

- **🎙️ Live Microphone Speech Recording**: Record voice directly in your browser on stage with one click.
- **⚡ ONNX Runtime Inference**: CPU-optimized inference supporting **16-bit Baseline**, **8-bit**, **4-bit**, **2-bit**, and **1-bit Binary** quantized NeMo Conformer models.
- **🗺️ 5 Kannada Dialect Regions**:
  - 🏙️ **South Karnataka** (Mysuru - Bengaluru)
  - 🌊 **Coastal Karnataka** (Mangaluru / Karavali)
  - 🏰 **North-West Karnataka** (Hubballi - Dharwad / Belagavi)
  - ☀️ **North-East Karnataka** (Kalaburagi / Hyderabad-Karnataka)
  - ⛰️ **Central Karnataka** (Shivamogga - Chitradurga)
- **📊 Real-time Analytics & Latency**: Real-time confidence probability distribution, acoustic waveform chart, and End-to-End Latency breakdown (Preprocess ms, Inference ms, Realtime Factor RTF).
- **🎵 Stage Presentation Fallback**: One-click demo audio snippets for each region in case microphone is unavailable or background noise is high on stage.

---

## 🚀 Quick Start (Stage Presentation Mode)

### Option 1: One-Click Launcher (Windows)
Double-click `run_app.bat` or run:
```cmd
.\run_app.bat
```

### Option 2: PowerShell Launcher
```powershell
.\run_app.ps1
```

### Option 3: Manual Activation & Run
1. Activate virtual environment:
```powershell
.\.venv\Scripts\Activate.ps1
```
2. Launch Streamlit:
```powershell
streamlit run Frontend\app.py
```

---

## 📂 Project Structure

```
Spire_app/
├── .venv/                      # Python Virtual Environment
├── Backend/
│   ├── Tlearn_Models/          # 14 ONNX Models (16bit, 8bit, 4bit, 2bit, 1bit)
│   ├── model_inference.py      # Audio Preprocessor & ONNX Inference Engine
│   ├── sample_audio.py         # Synthetic speech generator for demo clips
│   └── test_pipeline.py        # Pipeline test script
├── Frontend/
│   ├── app.py                  # Main Streamlit Cyberpunk UI
│   ├── components.py           # Custom UI cards, progress bars & metrics
│   └── styles.css              # Modern glassmorphism CSS theme
├── run_app.bat                 # One-click Windows batch launcher
├── run_app.ps1                 # One-click PowerShell launcher
└── README.md
```

---

## 🧠 Model Architecture & Quantization

| Model Variant | Quantization Scheme | Model Size | Expected Latency |
|---|---|---|---|
| `16bit_baseline.onnx` | FP16/FP32 Baseline | 59.5 MB | ~20-30 ms |
| `8bit_uniform.onnx` | 8-Bit Uniform | 59.5 MB | ~15-25 ms |
| `4bit_uniform.onnx` | 4-Bit Uniform | 59.5 MB | ~15-20 ms |
| `2bit_uniform.onnx` | 2-Bit Uniform | 59.5 MB | ~15-20 ms |
| `1bit_sign.onnx` | 1-Bit Binary Sign | 59.5 MB | ~15-20 ms |
