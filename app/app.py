import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------- Paths ----------
BASE = Path(__file__).resolve().parent.parent
MODELS = BASE / "models"
DATA = BASE / "data"

st.set_page_config(page_title="Churn Prediction", page_icon="📉", layout="wide")


# ---------- Model (training wala architecture, bilkul same) ----------
class ChurnANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


# ---------- Loaders ----------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load(MODELS / "scaler.pkl")
    feature_cols = joblib.load(MODELS / "feature_columns.pkl")
    config = joblib.load(MODELS / "config.pkl")
    model = ChurnANN(len(feature_cols))
    model.load_state_dict(torch.load(MODELS / "churn_ann.pt", map_location="cpu"))
    model.eval()
    return model, scaler, feature_cols, config


@st.cache_data
def load_data():
    df = pd.read_csv(DATA / "cleaned_churn.csv")
    results = pd.read_csv(MODELS / "results.csv")
    preds = pd.read_csv(MODELS / "test_predictions.csv")
    return df, results, preds


model, scaler, feature_cols, config = load_artifacts()
df, results, preds = load_data()
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def preprocess(input_dict):
    """Ek customer ki raw values ko model ke input me badalta hai."""
    row = pd.DataFrame([input_dict])
    combined = pd.concat([df.drop(columns="Churn"), row], ignore_index=True)
    enc = pd.get_dummies(combined, drop_first=True)
    x = enc.iloc[[-1]].reindex(columns=feature_cols, fill_value=0).astype("float32")
    x[NUM_COLS] = scaler.transform(x[NUM_COLS]).astype("float32")
    return torch.tensor(x.values, dtype=torch.float32)


# ---------- Sidebar ----------
st.sidebar.title("📉 Churn Prediction")
page = st.sidebar.radio("Page", ["1. Data Overview", "2. Model Comparison", "3. Live Prediction"])
st.sidebar.caption("Telco Customer Churn | ANN (PyTorch) vs ML baselines")


# ---------- Page 1: EDA ----------
if page == "1. Data Overview":
    st.title("Data Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churn rate", f"{df['Churn'].mean() * 100:.1f}%")
    c3.metric("Avg tenure (months)", f"{df['tenure'].mean():.1f}")

    st.subheader("Churn by Contract type")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.barplot(x="Contract", y="Churn", data=df, ax=ax)
    ax.set_ylabel("Churn rate")
    st.pyplot(fig)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Tenure vs Churn")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.boxplot(x="Churn", y="tenure", data=df, ax=ax)
        ax.set_xticklabels(["No", "Yes"])
        st.pyplot(fig)
    with col_b:
        st.subheader("Monthly Charges vs Churn")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.boxplot(x="Churn", y="MonthlyCharges", data=df, ax=ax)
        ax.set_xticklabels(["No", "Yes"])
        st.pyplot(fig)

    st.subheader("Data preview")
    st.dataframe(df.head(20), use_container_width=True)


# ---------- Page 2: Model Comparison ----------
elif page == "2. Model Comparison":
    st.title("Model Comparison")
    st.dataframe(results, use_container_width=True, hide_index=True)

    metric = st.selectbox("Metric chuno", ["ROC-AUC", "Recall", "Precision", "F1", "Accuracy"])
    st.bar_chart(results.set_index("Model")[metric])

    st.info(
        "ROC-AUC me sab models lagbhag barabar hain (~0.83-0.85). ANN ka high recall "
        "deep learning ki wajah se nahi, balkay class weighting aur threshold ki wajah se hai. "
        "Tabular data par simple models bhi utne hi acche hote hain."
    )


# ---------- Page 3: Live Prediction ----------
else:
    st.title("Live Prediction")

    st.subheader("Threshold explorer (test set par)")
    threshold = st.slider(
        "Decision threshold", 0.1, 0.9, float(config["default_threshold"]), 0.05,
        help="Kam threshold = zyada customers flag (high recall). Zyada threshold = kam, lekin zyada sahi flags (high precision).",
    )

    y_true = preds["y_true"].values
    y_hat = (preds["ann_prob"].values >= threshold).astype(int)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Precision", f"{precision_score(y_true, y_hat, zero_division=0):.3f}")
    m2.metric("Recall", f"{recall_score(y_true, y_hat):.3f}")
    m3.metric("F1", f"{f1_score(y_true, y_hat):.3f}")
    m4.metric("Accuracy", f"{accuracy_score(y_true, y_hat):.3f}")
    m5.metric("Flagged as high risk", f"{y_hat.mean() * 100:.1f}%")

    st.divider()
    st.subheader("Ek customer ka risk check karo")

    cat_cols = [c for c in df.select_dtypes(exclude="number").columns]
    inputs = {}

    c1, c2, c3 = st.columns(3)
    with c1:
        inputs["gender"] = st.selectbox("Gender", sorted(df["gender"].unique()))
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        inputs["SeniorCitizen"] = 1 if senior == "Yes" else 0
        inputs["Partner"] = st.selectbox("Partner", sorted(df["Partner"].unique()))
        inputs["Dependents"] = st.selectbox("Dependents", sorted(df["Dependents"].unique()))
        inputs["tenure"] = st.slider("Tenure (months)", 0, 72, 12)
    with c2:
        inputs["PhoneService"] = st.selectbox("Phone Service", sorted(df["PhoneService"].unique()))
        inputs["MultipleLines"] = st.selectbox("Multiple Lines", sorted(df["MultipleLines"].unique()))
        inputs["InternetService"] = st.selectbox("Internet Service", sorted(df["InternetService"].unique()))
        inputs["OnlineSecurity"] = st.selectbox("Online Security", sorted(df["OnlineSecurity"].unique()))
        inputs["OnlineBackup"] = st.selectbox("Online Backup", sorted(df["OnlineBackup"].unique()))
        inputs["DeviceProtection"] = st.selectbox("Device Protection", sorted(df["DeviceProtection"].unique()))
    with c3:
        inputs["TechSupport"] = st.selectbox("Tech Support", sorted(df["TechSupport"].unique()))
        inputs["StreamingTV"] = st.selectbox("Streaming TV", sorted(df["StreamingTV"].unique()))
        inputs["StreamingMovies"] = st.selectbox("Streaming Movies", sorted(df["StreamingMovies"].unique()))
        inputs["Contract"] = st.selectbox("Contract", sorted(df["Contract"].unique()))
        inputs["PaperlessBilling"] = st.selectbox("Paperless Billing", sorted(df["PaperlessBilling"].unique()))
        inputs["PaymentMethod"] = st.selectbox("Payment Method", sorted(df["PaymentMethod"].unique()))

    inputs["MonthlyCharges"] = st.number_input("Monthly Charges", 18.0, 120.0, 70.0, 0.5)
    inputs["TotalCharges"] = float(inputs["tenure"] * inputs["MonthlyCharges"])
    st.caption(f"Total Charges andaza se: tenure x monthly = {inputs['TotalCharges']:.1f}")

    if st.button("Predict", type="primary"):
        x = preprocess(inputs)
        with torch.no_grad():
            score = torch.sigmoid(model(x)).item()

        r1, r2 = st.columns(2)
        r1.metric("Churn risk score", f"{score * 100:.1f}%")
        if score >= threshold:
            r2.error(f"HIGH RISK (score {score:.2f} >= threshold {threshold:.2f})")
        else:
            r2.success(f"Low risk (score {score:.2f} < threshold {threshold:.2f})")
        st.progress(min(max(score, 0.0), 1.0))
        st.caption(
            "Ye 'risk score' hai, asli probability nahi: training me churn class ko extra weight diya gaya tha, "
            "isliye scores thore badhe hue hote hain."
        )