import streamlit as st

st.title("🛡️ Quantum Sentinel")
st.markdown("### Intrusion Detection in Quantum Networks using Machine Learning")
st.markdown(
    '**Final Projecy (ML Ops) "Microcredencial Introducción al Machine Learning"**'
)
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.header("The Challenge")
    st.markdown(
        """
    The **BB84 Quantum Key Distribution (QKD)** protocol is theoretically unbreakable due to the laws of physics.
    Any eavesdropper (Eve) trying to measure the photons alters their state, introducing errors.

    However, in practice:
    * Channels have **natural noise** (fiber optics, temperature).
    * Detectors have imperfections.

    So, the question is: **How can we distinguish between a natural error rate of 8% and a sophisticated attack by a spy trying to steal the key?**    
    """
    )

    st.info(
        "The classical solution is to set a fixed error threshold (e.g., 11%). If the error rate exceeds this, the transmission is discarded. But this is inefficient and vulnerable to subtle attacks that can operate just below the threshold."
    )

with col2:
    st.header("The Solution: ML")
    st.markdown(
        """
    The idea of **Quantum Sentinel** is to replace the fixed threshold with a **Supervised Machine Learning model**.

    Instead of just looking at the error rate, we analyze the full statistical behavior of the channel.
    
    **Project Objetives:**
    1. **Simulate** the BB84 protocol under various conditions (noise, attacks).
    2. **Train** a classifier (Random Forest) to detect attack patterns.
    3. **Deploy** a live monitoring system.
    """
    )

st.markdown("---")

# BB84 Protocol Explanation
st.header("Protocol BB84")

c_img, c_txt = st.columns([2, 1])

with c_img:
    st.image(
        image="images/bb84_attack_diagram.png",
        caption="Interception Scenario: BB84 Protocol with Eve's Interference",
        width="stretch",
    )

with c_txt:
    st.markdown("### How the attack works?")
    st.markdown(
        """
    1. **Alice** sends polarized photons to **Bob**.
    2. **Eve** tries to measure them. Without knowing the basis, she only gets it right 50% of the time.
    3. **Collapse:** When Eve chooses the wrong basis, the photon state collapses to a random state.
    4. **Bob** receives an altered photon.
    
    This process introduces errors in the coincidences **QBER (Quantum Bit Error Rate)**. If Eve is too aggressive, the error rate spikes, but if she's subtle, it can be hard to detect with a simple threshold.
    """
    )

st.markdown("---")

st.header("Proyect Overview")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.subheader("1. Exploration (EDA)")
    st.markdown(
        """
    * Generated dataset overview.
    * Visualization of the Overlap.
    * Correlation studies.
    """
    )

with col_b:
    st.subheader("2. The Model (ML)")
    st.markdown(
        """
    * Algorithm: **Random Forest**.
    * Performance Metrics.
    * Confusion Matrix.        
    """
    )

with col_c:
    st.subheader("3. Simulator (Live)")
    st.markdown(
        """
    * **Live Control Panel.**
    * Simulation of photon transmission.
    * Instant security diagnostics.
    """
    )

st.markdown("---")
st.caption(
    "Developed by Himar Edhey Hernández Alonso | Microcredencial Introducción al Machine Learning | 2026"
)
