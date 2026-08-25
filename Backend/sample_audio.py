import os
import numpy as np
import soundfile as sf
import io

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "sample_audios")

DEMO_SAMPLES = [
    {
        "id": 0,
        "filename": "sample_south_bengaluru.wav",
        "title": "South Karnataka (Bengaluru-Mysuru)",
        "phrase": "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ? (Namaskara, neevu hegeeddeeri?)",
        "dialect_id": 0,
        "base_freq": 180,
        "modulation": 4.0
    },
    {
        "id": 1,
        "filename": "sample_coastal_mangaluru.wav",
        "title": "Coastal Karnataka (Mangaluru/Karavali)",
        "phrase": "ಎಂಚ ಉಲ್ಲಾರ್? (Encha ullar? / How are you in Karavali dialect)",
        "dialect_id": 1,
        "base_freq": 220,
        "modulation": 7.0
    },
    {
        "id": 2,
        "filename": "sample_northwest_dharwad.wav",
        "title": "North-West Karnataka (Hubballi-Dharwad)",
        "phrase": "ಅರಾಮ್ ಇದೀರನ್ರಿ? (Aram ideeranri? / Dharwad dialect)",
        "dialect_id": 2,
        "base_freq": 160,
        "modulation": 9.0
    },
    {
        "id": 3,
        "filename": "sample_northeast_kalaburagi.wav",
        "title": "North-East Karnataka (Kalaburagi)",
        "phrase": "ಚಲೋ ಇದೀರಾ? (Chalo ideera? / Kalaburagi dialect)",
        "dialect_id": 3,
        "base_freq": 200,
        "modulation": 6.0
    },
    {
        "id": 4,
        "filename": "sample_central_shivamogga.wav",
        "title": "Central Karnataka (Shivamogga)",
        "phrase": "ಏನ್ ಸಮಾಚಾರ, ಚೆನ್ನಾಗಿದ್ದೀರಾ? (Een samachara, chennagiddeera?)",
        "dialect_id": 4,
        "base_freq": 190,
        "modulation": 5.0
    }
]

def generate_speech_like_waveform(base_freq=180, modulation=5.0, duration=3.0, sr=16000):
    """
    Generates a synthetic speech-like acoustic signal with formants, harmonic overtones,
    and syllable modulation patterns for demo audio fallback.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Syllable rhythm amplitude envelope (3-5 syllables per sec)
    syllable_env = 0.5 * (1.0 + np.sin(2 * np.pi * modulation * t))
    syllable_env = np.power(syllable_env, 1.5)
    
    # Formant frequency synthesis (F1, F2, F3)
    f0 = base_freq + 15.0 * np.sin(2 * np.pi * 1.5 * t)
    f1 = base_freq * 2.8
    f2 = base_freq * 7.2
    
    signal = (
        0.5 * np.sin(2 * np.pi * f0 * t) +
        0.3 * np.sin(2 * np.pi * f1 * t) +
        0.15 * np.sin(2 * np.pi * f2 * t)
    )
    
    # Add subtle breath noise
    noise = np.random.normal(0, 0.02, len(t))
    audio = (signal + noise) * syllable_env
    
    # Fade in & out
    fade_len = int(0.05 * sr)
    audio[:fade_len] *= np.linspace(0, 1, fade_len)
    audio[-fade_len:] *= np.linspace(1, 0, fade_len)
    
    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-5) * 0.8
    return audio.astype(np.float32)

def ensure_sample_audios():
    """Generates default demo WAV files in SAMPLES_DIR if not existing."""
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    generated = []
    
    for sample in DEMO_SAMPLES:
        file_path = os.path.join(SAMPLES_DIR, sample["filename"])
        if not os.path.exists(file_path):
            audio = generate_speech_like_waveform(
                base_freq=sample["base_freq"],
                modulation=sample["modulation"],
                duration=3.5,
                sr=16000
            )
            sf.write(file_path, audio, 16000)
            
        with open(file_path, "rb") as f:
            b = f.read()
            
        generated.append({
            "id": sample["id"],
            "title": sample["title"],
            "phrase": sample["phrase"],
            "filename": sample["filename"],
            "path": file_path,
            "bytes": b
        })
        
    return generated

if __name__ == "__main__":
    samples = ensure_sample_audios()
    print(f"Successfully ensured {len(samples)} sample audios!")
