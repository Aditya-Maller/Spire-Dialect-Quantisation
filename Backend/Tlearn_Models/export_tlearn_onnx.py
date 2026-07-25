import os
import gc
import torch
import torch.nn as nn
import nemo.collections.asr as nemo_asr
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cpu")

# ── 1. Define the True Transfer-Learning Architecture ────────────────────────
class NeMoDialectClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(NeMoDialectClassifier, self).__init__()
        # Load the 16-layer base model
        nemo_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_small")
        self.preprocessor = nemo_model.preprocessor
        self.encoder = nemo_model.encoder
        del nemo_model
        gc.collect()
        self.hidden_dim = 176
        # Single-layer classification head used during TL
        self.classifier = nn.Linear(self.hidden_dim, num_classes)

# ── 2. Wrapper to Bypass STFT Preprocessor for ONNX ──────────────────────────
class ConformerONNXWrapper(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, processed_signal, processed_length):
        # Taking pre-computed mel-spectrograms as input
        encoded, _ = self.encoder(audio_signal=processed_signal, length=processed_length)
        pooled = torch.mean(encoded, dim=-1)
        return self.classifier(pooled)

print("Instantiating base 16-layer TL model...")
base_model = NeMoDialectClassifier(num_classes=5).to(DEVICE)
onnx_model = ConformerONNXWrapper(base_model.encoder, base_model.classifier)
onnx_model.eval()

# ── 3. File Paths ─────────────────────────────────────────────────────────────
SERVER_ROOT = "/data/AdityaSMaller"
TRAINING_DIR = os.path.join(SERVER_ROOT, "training")
OUT_DIR = os.path.join(SERVER_ROOT, "Tlearn_Models")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = [
    {"name": "16bit_baseline", "path": f"{TRAINING_DIR}/baseline/model/baseline_conformer.pt"},
    {"name": "1bit_sign",      "path": f"{TRAINING_DIR}/1bit/Scheme1_Sign/model/scheme1_1bit_conformer.pt"},
    {"name": "2bit_uniform",   "path": f"{TRAINING_DIR}/2bit/Scheme2_Uniform/model/scheme2_2bit_conformer.pt"},
    {"name": "2bit_mulaw",     "path": f"{TRAINING_DIR}/2bit/Scheme3_MuLaw/model/scheme3_2bit_mulaw_conformer.pt"},
    {"name": "2bit_alaw",      "path": f"{TRAINING_DIR}/2bit/Scheme4_ALaw/model/scheme4_2bit_alaw_conformer.pt"},
    {"name": "2bit_log",       "path": f"{TRAINING_DIR}/2bit/Scheme5_Log/model/scheme5_2bit_log_conformer.pt"},
    {"name": "4bit_uniform",   "path": f"{TRAINING_DIR}/4bit/Scheme2_Uniform/model/scheme2_4bit_conformer.pt"},
    {"name": "4bit_mulaw",     "path": f"{TRAINING_DIR}/4bit/Scheme3_MuLaw/model/scheme3_4bit_mulaw_conformer.pt"},
    {"name": "4bit_alaw",      "path": f"{TRAINING_DIR}/4bit/Scheme4_ALaw/model/scheme4_4bit_alaw_conformer.pt"},
    {"name": "4bit_log",       "path": f"{TRAINING_DIR}/4bit/Scheme5_Log/model/scheme5_4bit_log_conformer.pt"},
    {"name": "8bit_uniform",   "path": f"{TRAINING_DIR}/8bit/Scheme2_Uniform/model/scheme2_8bit_conformer.pt"},
    {"name": "8bit_mulaw",     "path": f"{TRAINING_DIR}/8bit/Scheme3_MuLaw/model/scheme3_8bit_mulaw_conformer.pt"},
    {"name": "8bit_alaw",      "path": f"{TRAINING_DIR}/8bit/Scheme4_ALaw/model/scheme4_8bit_alaw_conformer.pt"},
    {"name": "8bit_log",       "path": f"{TRAINING_DIR}/8bit/Scheme5_Log/model/scheme5_8bit_log_conformer.pt"},
]

# ── 4. Export Loop ────────────────────────────────────────────────────────────
# Dummy input: 1 Batch, 80 Mel Features, 100 Time Frames
dummy_mels = torch.randn(1, 80, 100).to(DEVICE)
dummy_lengths = torch.tensor([100], dtype=torch.long).to(DEVICE)

for m in MODELS:
    print(f"\nProcessing {m['name']}...")
    if not os.path.exists(m['path']):
        print(f"  [ERROR] File not found: {m['path']}")
        continue

    # Load Weights into base_model
    chkpt = torch.load(m['path'], map_location=DEVICE)
    base_model.load_state_dict(chkpt['model_state_dict'])
    
    out_onnx = os.path.join(OUT_DIR, f"{m['name']}.onnx")
    print(f"  Exporting to {out_onnx}...")
    
    torch.onnx.export(
        onnx_model,
        (dummy_mels, dummy_lengths),
        out_onnx,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["mel_spectrogram", "sequence_length"],
        output_names=["dialect_logits"],
        dynamic_axes={
            "mel_spectrogram": {0: "batch_size", 2: "time_steps"},
            "sequence_length": {0: "batch_size"},
            "dialect_logits": {0: "batch_size"}
        }
    )
    print(f"  [SUCCESS] {m['name']} exported!")

print("\n🎉 All true Transfer-Learning models successfully exported!")
