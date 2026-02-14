import streamlit as st

st.set_page_config(layout="wide", page_title="Quantum Sentinel", page_icon="🛡️")

# Navegation Pages
# 1. Introduction
pg_intro = st.Page("views/intro.py", title="Introduction", icon="🏠")

# 2. EDA (Data Analysis)
pg_eda_raw = st.Page("views/eda/raw_data.py", title="Data Exploration", icon="📄")
pg_eda_viz = st.Page(
    "views/eda/visualization.py", title="Graphic Visualization", icon="📊"
)

# 3. Modeling (ML)
pg_model = st.Page("views/model/trainning.py", title="Model Metrics", icon="🧠")

# 4. Simulator (Live)
pg_demo = st.Page("views/app/simulator.py", title="Live Simulator", icon="🚀")

# Navegation System
pg = st.navigation(
    {
        "Project": [pg_intro],
        "Phase 1: Data Analysis (EDA)": [pg_eda_raw, pg_eda_viz],
        "Phase 2: Modeling ML": [pg_model],
        "Phase 3: Production": [pg_demo],
    }
)

pg.run()

# Sidebar Footer
with st.sidebar:
    st.divider()
    st.caption("🎓 Microcredencial Introduccion al ML - ULL")
    st.caption("Autor: Himar Edhey Hernández Alonso")
