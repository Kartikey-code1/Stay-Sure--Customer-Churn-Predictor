import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="StaySure Churn Predictor", page_icon="💚", layout="wide")

# ------------------ CSS ------------------
st.markdown("""
<style>
body {background-color: #0b0c10; color: #e6f7f1;}
.glow {color: #00fff2; text-shadow: 0 0 12px rgba(0,255,242,0.25); font-weight:700;}
.card {background: rgba(255,255,255,0.03); border-radius: 12px; padding: 16px; margin-bottom: 10px;}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown('<h1 class="glow" style="text-align:center;">💚 StaySure – Customer Churn Predictor</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered retention insights to prevent customer churn</p>", unsafe_allow_html=True)
st.markdown("---")

# ------------------ LOAD MODEL & ENCODERS ------------------
model_path = "models/churn_model.pkl"
encoder_path = "models/label_encoders.pkl"
feature_path = "models/feature_columns.pkl"

if not (os.path.exists(model_path) and os.path.exists(encoder_path) and os.path.exists(feature_path)):
    st.error("❌ Model files missing! Please run `python train_model.py` first.")
    st.stop()

model = joblib.load(model_path)
label_encoders = joblib.load(encoder_path)
feature_columns = joblib.load(feature_path)

# ------------------ PREPROCESS FUNCTION ------------------
def preprocess_input(df):
    df = df.copy()
    if "Churn" in df.columns:
        df = df.drop("Churn", axis=1)
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].map(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]
    return df

# ------------------ SIDEBAR CSV UPLOAD ------------------
st.sidebar.header("📂 Upload Customer CSV")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ CSV uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")

# ------------------ MAIN LAYOUT ------------------
col1, col2 = st.columns([1, 2])

# -------- Column 1: Single Predict --------
with col1:
    st.subheader("🔎 Quick Predict")
    tenure = st.slider("Tenure (months)", 0, 72, 24)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=2500.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
    tech_support = st.selectbox("TechSupport", ["Yes", "No"])
    payment_method = st.selectbox("PaymentMethod", ["Electronic check", "Credit card (automatic)", "Mailed check", "Bank transfer (automatic)"])

    if st.button("🔍 Predict Churn"):
        input_df = pd.DataFrame({
            "tenure": [tenure],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges],
            "Contract": [contract],
            "InternetService": [internet],
            "TechSupport": [tech_support],
            "PaymentMethod": [payment_method]
        })

        X_input = preprocess_input(input_df)
        proba = model.predict_proba(X_input)[:, 1][0]
        churn_percent = round(float(proba) * 100, 2)

        st.markdown(f"### 💡 Churn Probability: **{churn_percent}%**")
        if churn_percent > 60:
            st.error("⚠️ High Risk — Offer discounts or retention plan.")
        elif churn_percent > 30:
            st.warning("🟠 Moderate Risk — Engage customer.")
        else:
            st.success("✅ Low Risk — Loyal customer.")

# -------- Column 2: CSV Predictions --------
with col2:
    st.subheader("📊 Bulk Predictions (from CSV)")
    if df is not None:
        X = preprocess_input(df)
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
        df["ChurnPrediction"] = np.where(preds == 1, "Yes", "No")
        df["ChurnProbability(%)"] = (probs * 100).round(2)
        st.dataframe(df.head())

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions CSV", csv, "StaySure_Predictions.csv", "text/csv")
    else:
        st.info("Upload a CSV in sidebar to see bulk predictions.")

st.markdown("---")
st.markdown("<div class='card'><b>How to Use:</b><br>1️⃣ Place Telco CSV in <code>data/</code><br>2️⃣ Run <code>python train_model.py</code><br>3️⃣ Launch app with <code>streamlit run streamlit_app.py</code></div>", unsafe_allow_html=True)
