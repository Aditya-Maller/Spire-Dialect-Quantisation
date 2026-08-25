import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Backend.model_inference import run_dialect_inference, get_available_models
from Backend.sample_audio import ensure_sample_audios

if __name__ == "__main__":
    # Ensure Windows console handles UTF-8 output safely
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

    print("Testing ONNX Model Inference...")
    models = get_available_models()
    print(f"Found {len(models)} ONNX models:")
    for k, v in models.items():
        print(f"  - {k} ({v['size_mb']} MB, {v['badge']})")
        
    samples = ensure_sample_audios()
    sample = samples[0]
    
    baseline_model = os.path.join(os.path.dirname(__file__), "Tlearn_Models", "16bit_baseline.onnx")
    if os.path.exists(baseline_model):
        print(f"\nRunning inference on {baseline_model} using {sample['title']}...")
        result = run_dialect_inference(baseline_model, sample['bytes'])
        print("\n🎉 --- INFERENCE SUCCESSFUL --- 🎉")
        print(f"Predicted Dialect: {result['predicted_dialect']['name']}")
        print(f"Kannada Text: {result['predicted_dialect']['kannada']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Total Latency: {result['latency']['total_ms']} ms")
        print(f"Realtime Factor: {result['latency']['realtime_factor']}x")
    else:
        print("Model file not found!")
