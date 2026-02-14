import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from utils import FILE_PATH, MODEL_PATH


@st.cache_data
def load_data():
    try:
        df = pd.read_csv("quantum_security_data.csv")
        return df
    except FileNotFoundError as e:
        return None


def train_model(n_estimators=100, max_depth=5, criterion="gini"):
    """
    Entrena el modelo aceptando hiperparámetros dinámicos.
    """
    try:
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        return None, "Dataset not found."

    X = df[["qber", "sifted_count", "basis_match_rate"]]
    y = df["attacker_present"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        criterion=criterion,
        random_state=42,
    )

    clf.fit(X_train, y_train)
    joblib.dump(clf, MODEL_PATH)

    return clf


st.title("🧠 Model Tranning and Validation")
st.markdown(
    """
As we saw in the previous sections, the dataset generated from the quantum simulation 
contains an overlap between the two classes (Secure Channel vs Attack Detected), 
which makes it a non-trivial classification problem.

Because of this, we have chosen to implement a **Random Forest Classifier** for our 
model, which is well-suited for handling complex, non-linear relationships.
"""
)

df = load_data()

if df is None:
    st.error("Error: 'quantum_security_data.csv' not found.")
    st.warning("Please run the data generation before accessing this section.")
    st.page_link("views/eda/raw_data.py", label="Go to Data Generation", icon="⚙️")
    st.stop()


st.divider()
st.subheader("1. Model Management")

col_status, col_action = st.columns([2, 1])

try:
    _ = joblib.load(MODEL_PATH)
    model_exists = True
except FileNotFoundError:
    model_exists = False

with col_status:
    if model_exists is None:
        st.warning("Model not trained yet. Please train it.")
    else:
        st.success(f"Model found: `{MODEL_PATH}`")        

with st.expander("⚙️ Hyperparameters Configuration", expanded=not model_exists):
    st.markdown("Customize the Random Forest behavior:")

    c_h1, c_h2, c_h3 = st.columns(3)

    with c_h1:
        n_est = st.slider(
            "Number of Trees",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="More trees reduce variance (overfitting) but increase training time.",
        )

    with c_h2:
        depth = st.slider(
            "Max Depth",
            min_value=1,
            max_value=20,
            value=5,
            help="Deeper trees capture more details but can memorize noise (overfitting).",
        )

    with c_h3:
        crit = st.selectbox(
            "Criterion",
            options=["gini", "entropy", "log_loss"],
            index=0,
            help="The function to measure the quality of a split.",
        )

with col_action:
    btn_label = "🔄 Retrain Model" if model_exists else "⚙️ Train Model"

    if st.button(btn_label, type="primary" if not model_exists else "secondary"):
        with st.spinner("Training Random Forest Classifier..."):
            try:
                model = train_model(
                    n_estimators=n_est, 
                    max_depth=depth, 
                    criterion=crit
                )
                st.success("Model trained and saved successfully!")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Error during training: {e}")

if not model_exists:
    st.info("Click the button above to start the training process.")
    st.stop()

model = joblib.load(MODEL_PATH)

if model is None:
    st.error("Model training failed. Please check the dataset and try again.")
    st.stop()

X = df[["qber", "sifted_count", "basis_match_rate"]]
y = df["attacker_present"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_pred = model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.subheader("2. Performance Metrics (Test Set)")
st.markdown(
    """
Overview of model performance on the held-out test set.

- Accuracy: overall proportion of correct predictions.
- Precision: of the instances predicted as 'Attack', how many were true attacks.
- Recall (Sensitivity): of the real attacks, how many were detected by the model.
- F1-score: harmonic mean of Precision and Recall.
"""
)

st.caption(f"Evaluating {len(X_test)} samples that the model has never seen before.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{acc:.2%}", help="Overall, how often is the model correct?")
col2.metric(
    "Precision",
    f"{prec:.2%}",
    help="Of the instances predicted as 'Attack', how many were true attacks?",
)
col3.metric(
    "Recall (Sensibilidad)",
    f"{rec:.2%}",
    help="Of the real attacks, how many were detected by the model?",
)
col4.metric("F1-Score", f"{f1:.2%}", help="Precision y Recall balance")

if acc > 0.90:
    st.success(
        f"**Robust Model:** The Accuracy of {acc:.1%} significantly exceeds random chance (50%) and validates the project hypothesis."
    )
else:
    st.warning("The model needs tuning. Review the hyperparameters.")

st.divider()

# Results Analysis
# Confusion Matrix
st.subheader("3. Confusion Matrix")
st.markdown(
    """
The confusion matrix provides a detailed breakdown of the model's predictions:
- True Positives (TP): Attacks correctly identified.
- True Negatives (TN): Secure channels correctly identified.
- False Positives (FP): Secure channels incorrectly flagged as attacks.
- False Negatives (FN): Attacks that were missed (most critical error).
"""
)

col_izq, col_der = st.columns([3, 1])


with col_izq:
    cm = confusion_matrix(y_test, y_pred)

    labels = ["Secure", "Attack"]

    # Crear figura
    fig_cm, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
    )

    ax.set_ylabel("Reality")
    ax.set_xlabel("AI Prediction")
    ax.set_title("Where does the model make mistakes?")

    st.pyplot(fig_cm)

    st.info(
        """
    * **Reading:** * The dark blue diagonal are the correct predictions.
    * **False Negatives (Bottom-Left):** The most dangerous. Attacks that the we missed.
    """
    )
with col_der:
    st.markdown(
        """
    ### Analysis
    We can see that the model has identified most of the attacks correctly (TP), 
    but it has also missed a few attacks (FN), which is a concern. 

    The model is very good at identifying secure channels (TN), with less false alarms 
    (FP) than missed attacks (FN).  
    This suggests that while the model is effective, there is room for improvement in 
    reducing false negatives, which could be achieved through further tuning or by exploring
    more complex algorithms.
    """
    )


# Feature Importance

st.subheader("4. Feature Importance")
st.markdown(
    """The Random Forest model provides a measure of feature importance, which 
    indicates how much each feature contributes to the model's decision-making process.
    This can help us understand which physical parameters are most critical for detecting
    attacks in the BB84 protocol.
    """
)
col_izq, col_der = st.columns([3, 1])

with col_izq:
    importances = model.feature_importances_
    feature_names = X.columns

    df_imp = pd.DataFrame({"Variable": feature_names, "Importance": importances})
    df_imp = df_imp.sort_values("Importance", ascending=False)

    fig_imp, ax = plt.subplots(figsize=(6, 5))

    sns.barplot(
        data=df_imp,
        x="Importance",
        y="Variable",
        palette="viridis",
        ax=ax,
    )

    ax.set_title("Decision Influence")
    ax.set_xlabel("Relative Weight (0-1)")
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    st.pyplot(fig_imp)

with col_der:
    st.markdown(
        """
    ### Analysis:
    
    We can see that the model has assigned almost all the importance to the QBER 
    feature, which is consistent with our EDA findings.
    
    In the other hand, sifted_count and basis_match_rate have negligible importance,
    as we predicted based on the correlation analysis. This is a good sign, as it confirms 
    that the model has learned to focus on the most physically relevant feature for 
    detecting attacks in the BB84 protocol.
    """
    )

st.divider()

# Additional Algorithm Explanation
with st.expander("Algorithm Explanation", expanded=False):
    st.markdown(
        """
    * **Algorithm:** Random Forest Classifier (Scikit-Learn).
        * Random Forest was chosen for its ability to model non-linear decision boundaries
        and its robustness to the noise inherent in quantum systems.
        In essence, it builds an ensemble of decision trees and averages their predictions to
        improve generalization and reduce overfitting.
    * **Hyperparameters:**
        * `n_estimators`: 100 (number of decision trees).
        * `max_depth`: 5 (depth to prevent overfitting).
        * `criterion`: Gini Impurity.
    """
    )

# Download Model
with open(MODEL_PATH, "rb") as f:
    st.download_button(
        label="📥 Download the trained model (.pkl)",
        data=f,
        file_name=MODEL_PATH,
        mime="application/octet-stream",
    )
