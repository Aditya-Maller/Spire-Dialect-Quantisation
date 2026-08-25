import streamlit as st
import numpy as np

def render_header(model_info=None):
    """Renders the top presentation header with active status badges."""
    st.markdown("""
        <div>
            <div class="gradient-header">🎙️ SPIRE: NeMo Conformer Kannada Dialect AI</div>
            <div class="sub-header">Live Audio Speech Recognition & Dialect Classification (IISc RESPIN Corpus)</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Status badges
    model_name = model_info['name'] if model_info else "16bit_baseline.onnx"
    badge_label = model_info['badge'] if model_info else "16-BIT BASELINE"
    size_mb = model_info['size_mb'] if model_info else 59.5
    
    st.markdown(f"""
        <div style="margin-bottom: 20px;">
            <span class="status-badge badge-live">● STAGE PRESENTATION MODE</span>
            <span class="status-badge badge-quant">⚡ MODEL: {badge_label}</span>
            <span class="status-badge badge-cpu">💻 CPU INFERENCE ({size_mb} MB)</span>
        </div>
    """, unsafe_allow_html=True)

def render_predicted_result(result):
    """Renders the primary predicted dialect card."""
    dialect = result["predicted_dialect"]
    confidence = result["confidence"]
    duration = result["audio_duration_sec"]
    
    st.markdown(f"""
        <div class="result-card" style="border-color: {dialect['color']}80;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 2.8rem; margin-right: 10px;">{dialect['icon']}</span>
                    <span class="confidence-pill" style="background: {dialect['gradient']};">
                        {confidence}% Confidence
                    </span>
                </div>
                <div style="text-align: right; color: #94A3B8; font-size: 0.9rem;">
                    ⏱️ Audio Length: <b>{duration}s</b>
                </div>
            </div>
            
            <div style="margin-top: 15px;">
                <div class="result-title">{dialect['name']}</div>
                <div class="result-kannada">{dialect['kannada']}</div>
                <div style="color: #CBD5E1; font-size: 0.98rem; line-height: 1.5; margin-top: 8px;">
                    {dialect['description']}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_probability_bars(distribution):
    """Renders probability distribution progress bars for all 5 dialects."""
    st.markdown("### 📊 Dialect Classification Probabilities")
    
    for item in distribution:
        percent = item["percent"]
        color = item["color"]
        name = item["name"]
        kannada = item["kannada"]
        icon = item["icon"]
        
        st.markdown(f"""
            <div class="prob-item">
                <div class="prob-header">
                    <span>{icon} <b>{name}</b> <span style="color: #94A3B8; font-size: 0.85rem;">({kannada})</span></span>
                    <span style="color: {color}; font-weight: 700;">{percent}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width: {percent}%; background: {color};"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

def render_latency_metrics(latency, model_info):
    """Renders execution latency breakdown metrics."""
    st.markdown("### ⚡ Inference & Hardware Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{latency['preprocess_ms']} ms</div>
                <div class="metric-label">Audio Preprocess</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{latency['inference_ms']} ms</div>
                <div class="metric-label">ONNX Inference</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color: #00F2FE;">{latency['total_ms']} ms</div>
                <div class="metric-label">Total End-to-End</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col4:
        rtf = latency['realtime_factor']
        color = "#10B981" if rtf < 0.1 else "#F59E0B"
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color: {color};">{rtf}x</div>
                <div class="metric-label">Realtime Factor (RTF)</div>
            </div>
        """, unsafe_allow_html=True)

def render_waveform_chart(waveform_samples):
    """Renders acoustic audio waveform chart."""
    st.markdown("### 🌊 Audio Signal Waveform")
    st.line_chart(waveform_samples, height=120)

def render_quantization_overlay_chart(
    original_wf: np.ndarray,
    quantized_wf: np.ndarray,
    scheme_name: str,
    bit_depth: int,
    start_sec: float = 0.5,
    window_duration_ms: float = 40.0,
    sample_rate: int = 16000
):
    """
    Renders high-precision dual subplot comparing Original Speech vs Quantized Staircase Steps
    and the Quantization Residual Error e[t] = x[t] - x_q[t].
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Calculate time window indices
        start_idx = int(start_sec * sample_rate)
        num_samples = int((window_duration_ms / 1000.0) * sample_rate)
        end_idx = min(start_idx + num_samples, len(original_wf))
        
        if start_idx >= len(original_wf):
            start_idx = max(0, len(original_wf) - num_samples)
            end_idx = len(original_wf)
            
        t_sub = np.linspace(start_idx / sample_rate, end_idx / sample_rate, end_idx - start_idx) * 1000.0 # in ms
        orig_sub = original_wf[start_idx:end_idx]
        quant_sub = quantized_wf[start_idx:end_idx]
        error_sub = orig_sub - quant_sub

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            subplot_titles=(
                f"Acoustic Waveform Staircase: Original (Cyan) vs {scheme_name} {bit_depth}-bit (Magenta)",
                "Quantization Residual Error e[t] = Original[t] - Quantized[t]"
            ),
            row_heights=[0.7, 0.3]
        )

        # 1. Original 16-bit Waveform
        fig.add_trace(go.Scatter(
            x=t_sub,
            y=orig_sub,
            mode='lines',
            name='Original 16-bit Speech',
            line=dict(color='#00F2FE', width=2.5),
            opacity=0.9
        ), row=1, col=1)

        # 2. Quantized Staircase Waveform (line_shape='hv' creates distinct discrete steps!)
        fig.add_trace(go.Scatter(
            x=t_sub,
            y=quant_sub,
            mode='lines',
            line_shape='hv',
            name=f'{scheme_name} {bit_depth}-bit Steps',
            line=dict(color='#FF007F', width=2.2),
            opacity=0.95
        ), row=1, col=1)

        # Add unique discrete levels lines for low bit depths (<= 4 bits)
        if bit_depth <= 4 and scheme_name != "16-bit Baseline":
            unique_levels = np.unique(quant_sub)
            for lvl in unique_levels[:16]: # limit to 16 lines for readability
                fig.add_shape(
                    type="line",
                    x0=t_sub[0], x1=t_sub[-1],
                    y0=lvl, y1=lvl,
                    line=dict(color="rgba(255, 255, 255, 0.15)", width=1, dash="dot"),
                    row=1, col=1
                )

        # 3. Quantization Error Residual Subplot
        fig.add_trace(go.Scatter(
            x=t_sub,
            y=error_sub,
            mode='lines',
            name='Error Residual e[t]',
            line=dict(color='#FFD700', width=1.8),
            fill='tozeroy',
            fillcolor='rgba(255, 215, 0, 0.15)'
        ), row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(15, 23, 42, 0)',
            plot_bgcolor='rgba(22, 27, 38, 0.75)',
            font=dict(family="Outfit, sans-serif", color="#E2E8F0"),
            height=480,
            margin=dict(l=40, r=40, t=50, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified"
        )

        fig.update_xaxes(title_text="Time Position (milliseconds)", row=2, col=1, showgrid=True, gridcolor='rgba(255, 255, 255, 0.08)')
        fig.update_yaxes(title_text="Amplitude", row=1, col=1, showgrid=True, gridcolor='rgba(255, 255, 255, 0.08)', range=[-1.1, 1.1])
        fig.update_yaxes(title_text="Error e[t]", row=2, col=1, showgrid=True, gridcolor='rgba(255, 255, 255, 0.08)')

        st.plotly_chart(fig, use_container_width=True)

    except Exception as err:
        st.markdown(f"**Waveform Overlay ({scheme_name} {bit_depth}-bit)**")
        st.line_chart({
            "Original 16-bit": original_wf[::5],
            f"Quantized ({scheme_name})": quantized_wf[::5]
        }, height=300)
