import numpy as np

def sign_1bit_raw(x):
    return np.where(x >= 0, 1.0, -1.0)

def sign_1bit_gated(x, silence_thresh=0.02):
    out = np.where(x >= 0, 1.0, -1.0)
    # Mask out background silence so silent pauses stay 0.0!
    out[np.abs(x) < silence_thresh] = 0.0
    return out

if __name__ == "__main__":
    # Create speech signal (1 sec speech, 1 sec silence)
    sr = 16000
    t = np.linspace(0, 1.0, sr)
    speech = 0.5 * np.sin(2 * np.pi * 200 * t)
    silence = np.random.normal(0, 0.005, sr) # background mic noise
    audio = np.concatenate([speech, silence])
    
    raw_1bit = sign_1bit_raw(audio)
    gated_1bit = sign_1bit_gated(audio)
    
    print("Raw 1-bit energy in silence section:", np.mean(np.abs(raw_1bit[sr:]))) # 1.0 (LOUD NOISE!)
    print("Gated 1-bit energy in silence section:", np.mean(np.abs(gated_1bit[sr:]))) # 0.0 (CLEAN SILENCE!)
