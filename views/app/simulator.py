import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from utils import MODEL_PATH

st.title("🚀 Quantum Sentinel: Simulador en Vivo")
st.markdown(
    """
**Quantum Security Control Panel (QKD)**
This module simulates the transmission of photons through an optical channel and uses the AI model
to audit security in real time.
"""
)


@st.cache_resource
def load_ai_brain():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return None


model = load_ai_brain()

if model is None:
    st.error(
        f"Error: The AI model could not be loaded. Please ensure '{MODEL_PATH}' is in the correct directory."
    )
    st.stop()


# Simulation
def simulate_channel(n_bits, intercept_prob, noise_level):
    """
    Simulates a BB84 quantum key distribution scenario with optional eavesdropping and
    noise and returns the extracted features for AI analysis.
    """
    # Alice and Bob generation
    alice_bits = np.random.randint(0, 2, n_bits)
    alice_bases = np.random.randint(0, 2, n_bits)
    bob_bases = np.random.randint(0, 2, n_bits)
    bob_measurements = np.zeros(n_bits, dtype=int)

    # Quantum Channel
    for i in range(n_bits):
        bit = alice_bits[i]
        base_a = alice_bases[i]

        # Eve Interception
        if np.random.rand() < intercept_prob:
            eve_base = np.random.randint(0, 2)
            # If Eve chooses a different basis, she introduces an error
            if eve_base != base_a:
                bit = np.random.randint(0, 2)

        # Noise in the channel
        if np.random.rand() < noise_level:
            bit = 1 - bit  # Bit flip

        # Bob's measurement
        if base_a == bob_bases[i]:
            bob_measurements[i] = bit
        else:
            bob_measurements[i] = np.random.randint(0, 2)

    # Sifted Key Extraction
    match_indices = np.where(alice_bases == bob_bases)[0]

    if len(match_indices) == 0:
        return None

    sifted_alice = alice_bits[match_indices]
    sifted_bob = bob_measurements[match_indices]

    # Feature Extraction
    errors = np.sum(sifted_alice != sifted_bob)
    qber = errors / len(sifted_alice)
    basis_match_rate = len(sifted_alice) / n_bits

    return pd.DataFrame(
        {
            "qber": [qber],
            "sifted_count": [len(sifted_alice)],
            "basis_match_rate": [basis_match_rate],
        }
    )


# Control Panel (Sidebar)
with st.sidebar:
    st.header("Channel Parameters")

    st.subheader("Link Physics")
    n_bits = st.slider(
        "Sended Photons",
        128,
        2048,
        512,
        step=128,
        help="The longitude of the key to be transmitted.",
    )
    noise = st.slider(
        "Environmental Noise (%)",
        0.0,
        0.15,
        0.02,
        step=0.01,
        format="%.2f",
        help="Probability of noise in the channel.",
    )

    st.divider()

    st.subheader("Threat (Eva)")
    eve_active = st.toggle("Activate Interception", value=False)

    eve_aggressiveness = 0.0
    if eve_active:
        eve_aggressiveness = st.slider(
            "Aggressiveness (%)",
            0.1,
            1.0,
            0.5,
            help="Percentage of photons that Eve attempts to measure.",
        )
        st.warning("Eve is hearing the channel!")

    st.divider()

    simulate_btn = st.button(
        "Initiate Transmission", type="primary", use_container_width=True
    )

# Dashboard
if "simulation_done" not in st.session_state:
    st.session_state.simulation_done = False

if simulate_btn:
    st.session_state.simulation_done = True
    intercept_val = eve_aggressiveness if eve_active else 0.0
    features = simulate_channel(n_bits, intercept_val, noise)
    st.session_state.last_features = features

if st.session_state.simulation_done and st.session_state.last_features is not None:
    features = st.session_state.last_features

    # Prediction
    prediction = model.predict(features)[0]  # 0 o 1
    proba = model.predict_proba(features)[0]  # [Prob_0, Prob_1]

    qber_val = features["qber"].iloc[0]
    match_rate_val = features["basis_match_rate"].iloc[0]

    # Visualization
    if prediction == 1:
        st.error(f"**SECURITY ALERT: ATTACK DETECTED**")
        confidence = proba[1]
    else:
        st.success(f"**SECURE CHANNEL: NORMAL TRAFFIC**")
        confidence = proba[0]

    # Métricas clave en columnas
    m1, m2, m3 = st.columns(3)
    m1.metric("IA Confidence", f"{confidence:.2%}")
    m2.metric(
        "Current QBER",
        f"{qber_val:.2%}",
        delta=f"{qber_val-0.11:.2%}",
        delta_color="inverse",
    )
    m3.metric("Bits Sifted", f"{features['sifted_count'].iloc[0]}")

    st.divider()

    # 2. Gráficos Avanzados
    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.subheader("Error Mesaurement (QBER)")
        # Gauge Chart
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=qber_val * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Error Rate (%)"},
                delta={"reference": 11.0},  # Theoretical reference for BB84
                gauge={
                    "axis": {"range": [None, 30]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 11], "color": "lightgreen"},
                        {"range": [11, 30], "color": "lightcoral"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 11.0,
                    },
                },
            )
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.caption(
            "The red line (11%) marks the theoretical limit for unconditional security."
        )

    with c_right:
        st.subheader("Contextual Analysis")
        # Mockup with reference points for not loading real historical data
        df_ref = pd.DataFrame(
            {
                "qber": [0.02, 0.04, 0.15, 0.20],
                "rate": [0.5, 0.49, 0.5, 0.51],
                "tipo": ["Ref. Secure", "Ref. Secure", "Ref. Attack", "Ref. Attack"],
            }
        )

        tipo_actual = "ATTACK DETECTED" if prediction == 1 else "SECURE CHANNEL"

        fig_ctx = px.scatter(
            df_ref,
            x="qber",
            y="rate",
            color="tipo",
            color_discrete_map={"Ref. Secure": "green", "Ref. Attack": "red"},
            opacity=0.2,
        )  # Fondo transparente

        fig_ctx.add_trace(
            go.Scatter(
                x=[qber_val],
                y=[match_rate_val],
                mode="markers",
                marker=dict(
                    size=20,
                    color="red" if prediction == 1 else "green",
                    line=dict(width=2, color="black"),
                ),
                name=tipo_actual,
            )
        )

        fig_ctx.update_layout(
            title="Position in Phase Space",
            xaxis_title="QBER",
            yaxis_title="Basis Match Rate",
            showlegend=True,
        )
        st.plotly_chart(fig_ctx, use_container_width=True)

else:
    # Home Screen
    st.info("👈 Configure the parameters in the sidebar and click 'Start Simulation'.")
    st.image(
        image="images/quantum-sentinel.png",
        width="stretch",
    )
