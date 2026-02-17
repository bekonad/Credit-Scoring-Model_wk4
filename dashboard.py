import streamlit as st
import pandas as pd
import shap
import mlflow.sklearn
import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Bati Bank BNPL Credit Risk Tool",
    layout="wide",
    page_icon="💳"
)

st.title("Bati Bank Buy-Now-Pay-Later Credit Risk Tool")

st.markdown("""
**Week 12 Capstone – Production-Grade Credit Risk Scoring System**  

Random Forest classifier trained on engineered RFM and behavioral time features.  
Includes SHAP explainability for transparent risk assessment.
""")

st.info(
    "Model re-logged for correct artifact path. "
    "Features match training exactly (no CustomerId, no target leakage).",
    icon="ℹ️"
)

# =========================================================
# FEATURE DEFINITIONS (exact match to training)
# =========================================================
FEATURE_COLUMNS = [
    "total_amount",
    "avg_amount",
    "transaction_count",
    "std_amount",
    "avg_hour",
    "avg_day"
]

# =========================================================
# INPUT SECTION
# =========================================================
st.subheader("Customer Input Features")

col1, col2 = st.columns(2)

with col1:
    total_amount = st.slider(
        "Total Amount Spent (ETB)",
        -50000.0, 100000.0, 5000.0, 100.0
    )
    avg_amount = st.slider(
        "Average Transaction Amount (ETB)",
        -10000.0, 20000.0, 1000.0, 50.0
    )
    transaction_count = st.slider(
        "Transaction Count",
        1, 100, 10
    )

with col2:
    std_amount = st.slider(
        "Transaction Amount Volatility (Std Dev)",
        0.0, 10000.0, 500.0, 50.0
    )
    avg_hour = st.slider(
        "Average Transaction Hour",
        0.0, 23.0, 12.0, 1.0
    )
    avg_day = st.slider(
        "Average Transaction Day of Month",
        1.0, 31.0, 15.0, 1.0
    )

# =========================================================
# MODEL LOADING (cached)
# =========================================================
@st.cache_resource
def load_model():
    model_uri = "runs:/509e75e100aa43e99fafd4a978549b33/model"
    return mlflow.sklearn.load_model(model_uri)

# =========================================================
# MAIN CALCULATION
# =========================================================
model_loaded = False  # defined early

if st.button("Calculate Risk Score", type="primary"):

    input_data = pd.DataFrame([[
        total_amount,
        avg_amount,
        transaction_count,
        std_amount,
        avg_hour,
        avg_day
    ]], columns=FEATURE_COLUMNS)

    # Load model
    try:
        model = load_model()
        st.success("Random Forest model loaded successfully.")
        model_loaded = True
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.info("Using demo mode (simulated RFM-based risk score).")
        model_loaded = False

    # Prediction
    if model_loaded:
        input_data = input_data[model.feature_names_in_]
        prob = model.predict_proba(input_data)[0][1]
    else:
        # Demo fallback
        rfm_score = (transaction_count / 50.0) + (total_amount / 20000.0)
        volatility_penalty = std_amount / max(abs(total_amount), 1)
        time_score = (avg_hour / 24.0) + (avg_day / 31.0)
        combined = (rfm_score - volatility_penalty) * 0.7 + time_score * 0.3
        prob = max(0.05, min(0.90, 0.55 - combined * 0.45))

    score = max(300, min(850, int(850 - prob * 550)))

    # =========================================================
    # RESULTS DISPLAY
    # =========================================================
    st.subheader("Risk Assessment Results")

    colA, colB, colC = st.columns(3)

    colA.metric("Default Probability", f"{prob:.2%}")
    colB.metric("Credit Score (300–850)", score)
    colC.metric("Decision", "Decline" if prob > 0.35 else "Approve")

    if prob > 0.35:
        st.error("High Risk – Recommend declining or strict lending terms.")
    elif prob > 0.15:
        st.warning("Moderate Risk – Consider adjusted exposure.")
    else:
        st.success("Low Risk – Standard approval recommended.")

    # =========================================================
    # SHAP EXPLAINABILITY – FIXED WATERFALL FOR BINARY CLASSIFIER
    # =========================================================
    if model_loaded:
        st.subheader("Model Explanation (SHAP – High Risk Class)")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_data)  # shape: (1, n_features, 2 classes)

        # Select class 1 (high risk = is_high_risk == 1)
        shap_values_class1 = shap_values[:, :, 1]  # slice for positive class

        # Waterfall plot for class 1
        fig = plt.figure(figsize=(12, 8))
        shap.plots.waterfall(shap_values_class1[0], max_display=10)  # [0] for single sample
        st.pyplot(fig)

        # Base value info
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]

        st.caption(f"Base Probability for high risk class: {base_value:.2%}")
        st.caption("Red = feature increases predicted probability of high risk")
        st.caption("Blue = feature decreases predicted probability of high risk")
        st.caption("Top 10 features shown; waterfall starts from base value.")

    else:
        st.info("SHAP visualization requires the real model to be loaded.")

# =========================================================
st.markdown("---")
st.caption("10 Academy KAIM Week 12 Capstone | Credit Risk Modeling | © 2026")