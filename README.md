# 🛡️ Quantum Sentinel: QKD Security Auditor

**Intrusion detection in Quantum Key Distribution (QKD) networks using Machine Learning.**

This project simulates the **BB84** protocol, analyzes the statistical disturbances introduced by an eavesdropper (Eve), and trains a **Random Forest** model to classify the channel's security in real time, overcoming the limitations of static error thresholds.

---

## Overview

**Quantum Sentinel** is an interactive web application that demonstrates how Machine Learning can enhance the security of Quantum Key Distribution (QKD) systems. The project combines quantum physics simulation, data analysis, and ML to detect sophisticated eavesdropping attacks on the BB84 protocol.

The application provides:

- **Exploratory Data Analysis (EDA)** of quantum channel behavior
- **ML Model Training** with performance metrics
- **Live Simulator** for real-time intrusion detection

---

## The Challenge

The **BB84 Quantum Key Distribution (QKD)** protocol is theoretically unbreakable due to the laws of physics. Any eavesdropper (Eve) trying to measure the photons alters their state, introducing errors.

However, in practice:

- Channels have **natural noise** (fiber optics, temperature fluctuations)
- Detectors have imperfections
- Natural error rates can be similar to attack-induced errors

**Key Question**: How can we distinguish between a natural error rate of 8% and a sophisticated attack by an eavesdropper trying to steal the key?

The classical solution uses a fixed error threshold (e.g., 11% QBER). If the error rate exceeds this, the transmission is discarded. But this approach is not robust for real-world conditions where noise can fluctuate with small keys, needed, for example, for IoT devices that cannot handle the 512-bit keys usually required for error correction.

---

## The Solution

**Quantum Sentinel** replaces the fixed threshold with a **Supervised Machine Learning model** that analyzes the full statistical behavior of the quantum channel.

Instead of looking only at the QBER (Quantum Bit Error Rate), the model considers:

- **n_bits**: Number of transmitted bits
- **qber**: Quantum Bit Error Rate
- **sifted_count**: Number of matching basis measurements
- **basis_match_rate**: Proportion of matching bases
- **noise_level**: Channel noise intensity

This multi-dimensional analysis enables detection of subtle attack patterns that would evade traditional threshold-based systems.

---

## Installation and Usage

### How to run it on your own machine

1. **Clone the repository** (or download the files)

   ```bash
   cd quantum-sentinel
   ```

2. **Install the required dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**

   ```bash
   streamlit run app.py
   ```

The application will open in your default web browser at `http://localhost:8501`

### Navigation

The application is organized into four main sections:

1. **🏠 Introduction**: Overview of the project and BB84 protocol explanation
2. **📊 Phase 1: Data Analysis (EDA)**
   - Data Exploration: Raw dataset overview
   - Graphic Visualization: Statistical charts and correlations
3. **🧠 Phase 2: Modeling ML**: Model training, metrics, and feature importance
4. **🚀 Phase 3: Production**: Live simulator for real-time intrusion detection

---

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

---

## Author

**Himar Edhey Hernández Alonso**

🎓 Final Project - "Microcredencial Introducción al Machine Learning" - Universidad de La Laguna (ULL)
