import os
import sys
import numpy as np
import streamlit as st

# Add project root and Backend to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

backend_path = os.path.join(PROJECT_ROOT, "Backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from Backend.model_inference import (
    get_available_models,
    run_dialect_inference,
    preprocess_audio_bytes,
    DIALECT_INFO
)
from Backend.sample_audio import ensure_sample_audios
from Backend.quantization import (
    quantize_waveform,
    calculate_snr,
    calculate_mse,
    waveform_to_wav_bytes,
    normalise_waveform
)
from Frontend.components import (
    render_header,
    render_predicted_result,
    render_probability_bars,
    render_latency_metrics,
    render_waveform_chart,
    render_quantization_overlay_chart
)

# ── 1. Page Configuration ───────────────────────────────────────────────────
st.set_page_config(
    page_title="SPIRE - Kannada Dialect & Quantization AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
css_path = os.path.join(os.path.dirname(__file__), "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize Session State for Recorded Audio & Active Quantization
if "recorded_audio_bytes" not in st.session_state:
    st.session_state["recorded_audio_bytes"] = None
if "recorded_source" not in st.session_state:
    st.session_state["recorded_source"] = ""

# ── 2. Sidebar: Model Selection & Quantization Controls ─────────────────────
st.sidebar.markdown("## 🧠 Model Configuration")
available_models = get_available_models()

if not available_models:
    st.sidebar.error("No ONNX models found in `Backend/Tlearn_Models/` directory!")
    selected_model_key = None
    selected_model_info = None
else:
    model_options = list(available_models.keys())
    default_index = 0
    for idx, key in enumerate(model_options):
        if "16bit" in key or "baseline" in key:
            default_index = idx
            break
            
    selected_model_key = st.sidebar.selectbox(
        "Select Model Checkpoint:",
        options=model_options,
        index=default_index,
        format_func=lambda k: f"{available_models[k]['badge']} ({available_models[k]['size_mb']} MB)"
    )
    selected_model_info = available_models[selected_model_key]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Active Model:** `{selected_model_info['name']}`")
    st.sidebar.markdown(f"**Quantization:** {selected_model_info['type']}")
    st.sidebar.markdown(f"**File Size:** `{selected_model_info['size_mb']} MB`")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🗺️ Kannada Dialect Regions")
for d_id, d_data in DIALECT_INFO.items():
    st.sidebar.markdown(f"{d_data['icon']} **{d_data['name']}**")

st.sidebar.markdown("---")
st.sidebar.caption("SPIRE Speech & Language Processing Lab | NeMo Conformer ONNX Runtime")

# ── 3. Main Header ──────────────────────────────────────────────────────────
render_header(selected_model_info)

demo_samples = ensure_sample_audios()

# Automatically set default recorded audio to demo sample 0 if empty
if st.session_state["recorded_audio_bytes"] is None and len(demo_samples) > 0:
    st.session_state["recorded_audio_bytes"] = demo_samples[0]["bytes"]
    st.session_state["recorded_source"] = f"Demo Sample: {demo_samples[0]['title']}"

# ── 4. Main Navigation Tabs ─────────────────────────────────────────────────
main_tab1, main_tab2 = st.tabs([
    "🎙️ Live Speech & Model Classification",
    "🎛️ Interactive Quantization Visualizer & Audio Simulator"
])

# ============================================================================
# TAB 1: Live Speech Recording & Model Inference
# ============================================================================
with main_tab1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs([
        "🎙️ Live Mic Recording",
        "📁 Upload Audio File",
        "🎵 Demo Region Samples"
    ])

    with tab1:
        st.markdown("#### Record Voice Live on Stage")
        st.caption("Click the microphone button below to record your voice live during presentation.")
        
        try:
            from audio_recorder_streamlit import audio_recorder
            col_rec1, col_rec2 = st.columns([1, 3])
            with col_rec1:
                st.markdown("<div class='mic-wrapper'>", unsafe_allow_html=True)
                mic_bytes = audio_recorder(
                    text="Click Mic to Record",
                    recording_color="#e83e8c",
                    neutral_color="#00f2fe",
                    icon_size="2x"
                )
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_rec2:
                if mic_bytes:
                    st.success("✅ Live Audio Recorded Successfully!")
                    st.audio(mic_bytes, format="audio/wav")
                    st.session_state["recorded_audio_bytes"] = mic_bytes
                    st.session_state["recorded_source"] = "Live Microphone Recording"
                else:
                    st.info("💡 Press the blue microphone button to start recording. Speak for 2-4 seconds in Kannada.")
        except Exception as e:
            st.warning("Standard Streamlit audio recorder initialized.")
            uploaded_mic = st.file_uploader("Record or select mic file:", type=["wav", "mp3", "m4a"], key="mic_fallback")
            if uploaded_mic:
                st.session_state["recorded_audio_bytes"] = uploaded_mic.read()
                st.session_state["recorded_source"] = "Uploaded Audio Recording"

    with tab2:
        st.markdown("#### Upload Speech Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file (.wav, .mp3, .flac)",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            key="file_upload"
        )
        if uploaded_file is not None:
            st.session_state["recorded_audio_bytes"] = uploaded_file.read()
            st.session_state["recorded_source"] = f"File: {uploaded_file.name}"
            st.audio(st.session_state["recorded_audio_bytes"])

    with tab3:
        st.markdown("#### One-Click Demo Dialect Audio Clips")
        st.caption("Instant test audio clips for each Kannada dialect region for presentation reliability.")
        
        cols = st.columns(len(demo_samples))
        for idx, sample in enumerate(demo_samples):
            with cols[idx]:
                d_info = DIALECT_INFO[sample['id']]
                st.markdown(f"**{d_info['icon']} {sample['title'].split('(')[0]}**")
                st.caption(f"_{sample['phrase']}_")
                st.audio(sample['bytes'], format="audio/wav")
                if st.button(f"Predict {d_info['icon']}", key=f"btn_demo_{idx}", use_container_width=True):
                    st.session_state["recorded_audio_bytes"] = sample['bytes']
                    st.session_state["recorded_source"] = f"Demo Sample: {sample['title']}"

    st.markdown("</div>", unsafe_allow_html=True)

    # Run Prediction & Render Results if audio is ready
    if st.session_state["recorded_audio_bytes"] is not None and selected_model_info is not None:
        st.markdown("---")
        st.markdown(f"### ⚡ Running Model Inference (`{selected_model_info['name']}`)...")
        
        with st.spinner("Processing audio & predicting Kannada dialect..."):
            try:
                result = run_dialect_inference(selected_model_info["path"], st.session_state["recorded_audio_bytes"])
                
                # Primary Predicted Dialect Banner
                render_predicted_result(result)
                
                # Two Column Layout: Probability Bars + Signal Waveform
                col_left, col_right = st.columns([3, 2])
                
                with col_left:
                    render_probability_bars(result["distribution"])
                    
                with col_right:
                    render_waveform_chart(result["waveform_sample"])
                    
                st.markdown("---")
                # Hardware & Latency Metrics
                render_latency_metrics(result["latency"], selected_model_info)
                
            except Exception as err:
                st.error(f"Error during model inference: {str(err)}")
                st.exception(err)
                
    elif selected_model_info is None:
        st.warning("Please ensure model `.onnx` files are located in `Backend/Tlearn_Models/`.")
    else:
        st.info("👈 Record live audio, upload a file, or click a Demo Sample above to start dialect prediction.")

# ============================================================================
# TAB 2: Interactive Acoustic Quantization Visualizer & Audio Simulator
# ============================================================================
with main_tab2:
    st.markdown("## 🎛️ Acoustic Quantization Visualizer & Audio Simulator")
    st.caption("Inspect how quantization schemes discretize your recorded speech waveform into staircase levels and listen to the audio playback live!")

    if st.session_state["recorded_audio_bytes"] is None:
        st.warning("⚠️ No audio recorded yet! Please record voice or select a demo audio sample in Tab 1 first.")
    else:
        st.markdown(f"**Active Audio Source:** `{st.session_state['recorded_source']}`")
        
        # Decode original audio to float waveform
        orig_waveform_tensor = preprocess_audio_bytes(st.session_state["recorded_audio_bytes"])
        orig_waveform = orig_waveform_tensor.squeeze(0).numpy()
        orig_norm, orig_peak = normalise_waveform(orig_waveform)
        total_duration_sec = len(orig_waveform) / 16000.0
        
        # Quantization Controls Card
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        col_q1, col_q2, col_q3 = st.columns([2, 2, 2])
        
        with col_q1:
            q_scheme = st.selectbox(
                "Select Quantization Scheme:",
                options=["1-bit Sign", "1-bit On-Off Pulse", "Uniform", "Mu-law", "A-law", "Logarithmic", "16-bit Baseline"],
                index=0,
                help="1-bit Sign: Zero-crossing binary sign (+1/-1)\n1-bit On-Off Pulse: Threshold pulse (Morse code style)\nUniform: Mid-Tread Linear PCM\nMu-law: ITU-T G.711 Companding & Expansion (mu=255)\nA-law: ITU-T G.711 Companding & Expansion (A=87.6)\nLogarithmic: Base-2 quantization\n16-bit: Full precision reference"
            )
            
        with col_q2:
            if "1-bit" in q_scheme:
                q_bit_depth = 1
                st.markdown("**Bit Depth:** `1-bit` (Fixed)")
            elif q_scheme == "16-bit Baseline":
                q_bit_depth = 16
                st.markdown("**Bit Depth:** `16-bit` (Full Precision)")
            else:
                q_bit_depth = st.select_slider(
                    "Select Target Bit Depth:",
                    options=[1, 2, 4, 8, 16],
                    value=2,
                    format_func=lambda b: f"{b}-bit ({2**b} levels)"
                )
                
        with col_q3:
            enable_gate = st.checkbox("🔇 Silence Noise Gate (Mute mic noise during pauses)", value=True)
            gate_threshold = st.slider("Silence Threshold:", 0.005, 0.05, 0.02, 0.005) if enable_gate else 0.0
                
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Apply Quantization Scheme to Waveform
        quantized_wf = quantize_waveform(
            orig_norm,
            q_scheme,
            q_bit_depth,
            silence_gate=enable_gate,
            silence_thresh=gate_threshold
        )
        
        # Calculate Signal Metrics
        snr_db = calculate_snr(orig_norm, quantized_wf)
        mse_val = calculate_mse(orig_norm, quantized_wf)
        bit_reduction = round((1.0 - (q_bit_depth / 16.0)) * 100.0, 1)
        
        # Render Metrics Banner
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value" style="color: #00F2FE;">{snr_db} dB</div>
                    <div class="metric-label">Signal-to-Noise Ratio (SNR)</div>
                </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{mse_val}</div>
                    <div class="metric-label">Mean Squared Error (MSE)</div>
                </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value" style="color: #10B981;">{bit_reduction}%</div>
                    <div class="metric-label">Data Reduction vs 16-bit</div>
                </div>
            """, unsafe_allow_html=True)
        with m_col4:
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value" style="color: #C084FC;">{q_bit_depth} bits</div>
                    <div class="metric-label">Active Bit Depth</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # Interactive Plot Controls for Zooming & Time Inspection
        col_ctrl1, col_ctrl2 = st.columns([3, 2])
        with col_ctrl1:
            time_start_pos = st.slider(
                "⏱️ Select Time Position for Inspection (seconds):",
                min_value=0.0,
                max_value=max(0.0, total_duration_sec - 0.05),
                value=min(0.5, max(0.0, total_duration_sec - 0.05)),
                step=0.05
            )
        with col_ctrl2:
            zoom_win_ms = st.select_slider(
                "🔍 Zoom Level / Window Duration:",
                options=[10, 20, 40, 80, 150, 300],
                value=40,
                format_func=lambda ms: f"{ms} ms window"
            )
        
        # 1. Interactive Plotly Waveform Overlay
        render_quantization_overlay_chart(
            orig_norm,
            quantized_wf,
            q_scheme,
            q_bit_depth,
            start_sec=time_start_pos,
            window_duration_ms=zoom_win_ms,
            sample_rate=16000
        )
        
        st.markdown("---")
        
        # 2. Audio Playback & Comparison Panel (Crisp Volume Scale Preserved)
        col_aud1, col_aud2 = st.columns(2)
        
        with col_aud1:
            st.markdown("### 🔊 Original 16-bit Speech Audio")
            orig_wav_bytes = waveform_to_wav_bytes(orig_norm, 16000, orig_peak)
            st.audio(orig_wav_bytes, format="audio/wav")
            
        with col_aud2:
            st.markdown(f"### 📻 Quantized Audio ({q_scheme} {q_bit_depth}-bit)")
            quantized_wav_bytes = waveform_to_wav_bytes(quantized_wf, 16000, orig_peak)
            st.audio(quantized_wav_bytes, format="audio/wav")
            
            st.download_button(
                label=f"⬇️ Download {q_scheme} {q_bit_depth}-bit WAV",
                data=quantized_wav_bytes,
                file_name=f"quantized_{q_scheme.lower().replace(' ', '_')}_{q_bit_depth}bit.wav",
                mime="audio/wav"
            )
            
        st.markdown("---")
        
        # 3. Model Classification on Quantized Waveform
        if selected_model_info is not None:
            st.markdown(f"### 🤖 NeMo Conformer Model Prediction on Quantized Audio")
            st.caption("Runs your fine-tuned NeMo Conformer model directly on the quantized speech audio to evaluate model robustness.")
            
            if st.button("⚡ Test Model Prediction on Quantized Waveform", use_container_width=True, type="primary"):
                with st.spinner("Running model on quantized audio..."):
                    try:
                        q_result = run_dialect_inference(selected_model_info["path"], quantized_wav_bytes)
                        render_predicted_result(q_result)
                        render_probability_bars(q_result["distribution"])
                    except Exception as q_err:
                        st.error(f"Error running model on quantized audio: {str(q_err)}")
