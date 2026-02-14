import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Graphic Visualization")
st.markdown(
    """
Now, we are going to visualize the dataset we generated in the previous step. This is a
crucial part of the EDA process, as it allows us to understand the underlying structure 
of the data and identify patterns that may not be immediately obvious from raw numbers.

Using standard engineering libraries (Matplotlib/Seaborn), we will analyze the 
separability of the classes. The goal is to identify if there is a clear boundary 
between **Natural Noise** and an **Eve's Attack**.
"""
)


@st.cache_data
def load_data():
    try:
        df = pd.read_csv("quantum_security_data.csv")
        df["Tag"] = df["attacker_present"].map(
            {0: "Secure Channel (Noise)", 1: "Attack Detected"}
        )
        return df
    except FileNotFoundError:
        return None


df = load_data()

if df is None:
    st.error(
        "Error: 'quantum_security_data.csv' not found. Please run the data generation notebook first."
    )
    st.stop()

#
st.subheader("1. The Overlap: QBER Distribution")
st.markdown(
    """
As we saw in the previous section, the Quantum Bit Error Rate (QBER) variable was the 
most correlated with the target variable. Now, let's visualize how the QBER values are
distributed for both classes (Secure Channel vs Attack Detected) using an histogram.
"""
)

col1, col2 = st.columns([3, 1])

with col1:
    fig_hist, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data=df,
        x="qber",
        hue="Tag",  # Color by class
        element="step",
        stat="density",  # Using density to compare shapes
        common_norm=False,
        palette={"Secure Channel (Noise)": "green", "Attack Detected": "red"},
        alpha=0.5,
        ax=ax,
    )

    # Theoretical limit line (11%)
    ax.axvline(
        x=0.11,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Theoretical Threshold (11%)",
    )
    ax.set_title("Distribution of QBER: Critical Overlap")
    ax.set_xlabel("Error Rate (QBER)")
    ax.legend()

    st.pyplot(fig_hist)
with col2:
    st.markdown(
        """
    ### Analysis
    The green area (Secure Channel) and red area (Attack Detected) overlap significantly 
    between 0% and 15% QBER (aprox). Any fixed threshold in this region would lead to 
    misclassifications, highlighting the need for a more sophisticated detection method (ML).
    """
    )

st.divider()

st.subheader("2. Correlation Matrix")

st.markdown(
    """We done yet the univariate analysis. Now, let's see how the features interact 
    with each other and with the target variable. We only include numerical features in
    the correlation matrix.
    """
)

col1, col2 = st.columns([3, 1])

with col1:
    cols_num = ["qber", "sifted_count", "basis_match_rate", "attacker_present"]
    corr_matrix = df[cols_num].corr()

    fig_corr, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_title("Pearson Correlation Matrix")
    st.pyplot(fig_corr)

with col2:
    st.markdown(
        """
    ### Analysis
    The correlation matrix confirms that QBER has the strongest positive correlation with 
    the target variable (attacker_present), while Basis Match Rate has almost null  
    correlation. This suggests that while QBER is the most important predictor, there may 
    be complex interactions that a simple threshold cannot capture.
    """
    )


st.divider()

st.subheader("3. Multivariable Analysis: QBER vs Basis Match Rate")
st.markdown(
    """
    Before, we only looked at QBER. But adding another feature like could provide 
    additional context.
    
    As we could see in the correlation matrix, Basis Match Rate and shifted_count are 
    essentially the same variable (they are perfectly correlated), so we will only use 
    Basis Match Rate for the visualization. Checking is this time is possible separate 
    the classes with a simple linear boundary.
    """
)

col1, col2 = st.columns([3, 1])

with col1:
    fig_scatter, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x="qber",
        y="basis_match_rate",
        hue="Tag",
        style="Tag",  # Diferente forma de punto para daltónicos
        palette={"Secure Channel (Noise)": "green", "Attack Detected": "red"},
        alpha=0.6,
        s=60,  # Tamaño de los puntos
        ax=ax,
    )

    ax.set_title("Features Space")
    ax.set_xlabel("QBER")
    ax.set_ylabel("Basis Match Rate (%)")
    ax.grid(True, linestyle="--", alpha=0.5)

    st.pyplot(fig_scatter)

with col2:
    st.markdown(
        """
    ### Analysis
    We can see that despite adding the Basis Match Rate, there is still a significant 
    overlap between the two classes, so their are not separable by a simple linear boundary.

    Although QBER is the most important predictor, there is a complex overlap at low
    error rates. So linear classifiers as Logistic Regression would struggle to find a 
    good decision boundary. This justifies the use of Machine Learning models that can
    capture non-linear patterns and interactions between features. 
    """
    )

st.success(
    """
## EDA Conclusion: 
1. **QBER** is the most critical feature, but it alone cannot perfectly separate the classes due to significant overlap.
2. **Basis Match Rate** has a weaker correlation but may provide additional context for the ML.
 
In clonclusion, the visualizations confirm that the problem of distinguishing between 
natural noise and an attack is non-trivial. The significant overlap in QBER distributions
and the complex feature interactions highlight the limitations of traditional 
threshold-based methods, reinforcing the need for a sophisticated ML approach.
"""
)
