import os
import json
import torch
import torchaudio
import numpy as np
import pandas as pd
import onnxruntime as ort
import nemo.collections.asr as nemo_asr
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ── Paths & Setup ─────────────────────────────────────────────────────────────
BASE_DATA_DIR = "/data/AdityaSMaller/Database/RESPIN/audios/kannada"
TEST_DIR = os.path.join(BASE_DATA_DIR, "IISc_RESPIN_test_kn")
TEST_META_PATH = os.path.join(TEST_DIR, "meta_test_kn.json")
TRAIN_META_PATH = os.path.join(BASE_DATA_DIR, "IISc_RESPIN_train_kn_s5", "meta_IISc_RESPIN_train_kn_s5.json")

# Load Metadata to establish consistent dialect-to-ID mapping
print("Loading dataset metadata...")
with open(TRAIN_META_PATH, "r", encoding="utf-8") as f:
    train_meta = pd.DataFrame.from_dict(json.load(f), orient="index")
with open(TEST_META_PATH, "r", encoding="utf-8") as f:
    test_meta = pd.DataFrame.from_dict(json.load(f), orient="index")

dialects = sorted(train_meta["dialect"].unique())
dialect_to_id = {d: i for i, d in enumerate(dialects)}

# ── Dataset & DataLoader ──────────────────────────────────────────────────────
class RawRespinDataset(Dataset):
    def __init__(self, meta_df, source_dir, dialect_map):
        self.paths = [os.path.join(source_dir, wp) for wp in meta_df["wav_path"]]
        self.labels = [dialect_map[d] for d in meta_df["dialect"]]

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        try:
            waveform, sr = torchaudio.load(path)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            return waveform.squeeze(0), label
        except Exception:
            # Return silence if unreadable
            return torch.zeros(16000), label

def pad_collate_fn(batch):
    waveforms, labels = zip(*batch)
    lengths = torch.tensor([len(w) for w in waveforms], dtype=torch.long)
    padded_waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_waveforms, lengths, labels

# Batch size 32 is fast on ONNX Runtime CPU
test_dataset = RawRespinDataset(test_meta, TEST_DIR, dialect_to_id)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=pad_collate_fn)

# ── Preprocessor ──────────────────────────────────────────────────────────────
print("Loading NeMo Audio Preprocessor...")
nemo_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_small")
preprocessor = nemo_model.preprocessor
preprocessor.eval()
# Keep on CPU because ONNX inference will run on CPU
device = torch.device("cpu") 
preprocessor.to(device)

# ── ONNX Inference Loop ───────────────────────────────────────────────────────
TLEARN_DIR = "/data/AdityaSMaller/Tlearn_Models"
onnx_files = [f for f in os.listdir(TLEARN_DIR) if f.endswith(".onnx")]
print(f"\nFound {len(onnx_files)} ONNX models to evaluate.")

for onnx_file in sorted(onnx_files):
    model_path = os.path.join(TLEARN_DIR, onnx_file)
    print(f"\nEvaluating: {onnx_file}")
    
    # Initialize ONNX Runtime Session
    # Using default CPUExecutionProvider
    ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for waveforms, lengths, labels in tqdm(test_loader, desc="Inference", leave=False):
            waveforms = waveforms.to(device)
            lengths = lengths.to(device)
            
            # 1. PyTorch Mel Spectrogram Extraction
            processed_signal, processed_length = preprocessor(input_signal=waveforms, length=lengths)
            
            # 2. Convert to numpy for ONNX
            mel_np = processed_signal.numpy()
            len_np = processed_length.numpy()
            
            # 3. ONNX Graph Execution
            ort_inputs = {
                "mel_spectrogram": mel_np,
                "sequence_length": len_np
            }
            logits = ort_session.run(["dialect_logits"], ort_inputs)[0]
            
            # 4. Score
            preds = np.argmax(logits, axis=1)
            correct += np.sum(preds == labels.numpy())
            total += len(labels)
            
    accuracy = correct / total
    print(f"==> Accuracy for {onnx_file}: {accuracy * 100:.2f}%")
