import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("🚨 Fraud Detection Dashboard")
st.markdown("Model: XGBoost + SMOTE")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load("model_xgb_smote.pkl")

model = load_model()

# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess(df):
    df = df.copy()
    
    # Feature Engineering (HARUS sama dengan training)
    if 'Time' in df.columns:
        df['hour'] = (df['Time'] // 3600) % 24
    if 'Amount' in df.columns:
        df['Amount_log'] = np.log1p(df['Amount'])
    
    # Drop original
    df = df.drop(columns=['Time', 'Amount'], errors='ignore')
    
    return df

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Settings")

threshold = st.sidebar.slider("Threshold Fraud", 0.0, 1.0, 0.3)
top_n = st.sidebar.slider("Top N Fraud", 5, 50, 10)

st.sidebar.write("Model:", type(model).__name__)

# =========================
# LOAD DATA
# =========================
uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Menggunakan sample data")
    df = pd.read_csv("creditcard.csv").sample(5000, random_state=42)

st.subheader("📄 Data Preview")
st.dataframe(df.head())

# =========================
# PREPROCESS
# =========================
df_clean = preprocess(df)

X = df_clean.drop(columns=['Class'], errors='ignore')

# =========================
# HANDLE FEATURE MISMATCH
# =========================
expected_features = model.feature_names_in_

# Tambahkan kolom yang hilang
missing_cols = set(expected_features) - set(X.columns)
for col in missing_cols:
    X[col] = 0

# Pastikan urutan sama
X = X[expected_features]

# =========================
# PREDICTION
# =========================
y_prob = model.predict_proba(X)[:, 1]

df['fraud_prob'] = y_prob
df['prediction'] = (df['fraud_prob'] >= threshold).astype(int)

# =========================
# METRICS
# =========================
st.subheader("📊 Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Total Data", len(df))
col2.metric("Predicted Fraud", int(df['prediction'].sum()))
col3.metric("Avg Fraud Probability", round(df['fraud_prob'].mean(), 4))

# =========================
# TOP FRAUD
# =========================
st.subheader(f"🔥 Top {top_n} Transaksi Berisiko")

top_df = df.sort_values("fraud_prob", ascending=False).head(top_n)
st.dataframe(top_df)

# =========================
# DISTRIBUTION
# =========================
st.subheader("📈 Distribusi Probabilitas Fraud")
st.bar_chart(df['fraud_prob'].head(100))

# =========================
# HIGH RISK
# =========================
st.subheader("🔍 Transaksi Risiko Tinggi")

high_risk = df[df['fraud_prob'] >= threshold]

st.write(f"Jumlah transaksi berisiko tinggi: {len(high_risk)}")
st.dataframe(high_risk.head(20))

# =========================
# FEATURE IMPORTANCE
# =========================
st.subheader("📊 Feature Importance")

if hasattr(model, "feature_importances_"):
    importance = model.feature_importances_
    features = expected_features

    # ambil top 10 biar rapi
    idx = np.argsort(importance)[-10:]

    fig, ax = plt.subplots()
    ax.barh(np.array(features)[idx], importance[idx])
    st.pyplot(fig)
else:
    st.info("Model tidak mendukung feature importance")

# =========================
# INSIGHT
# =========================
st.subheader("🧠 Insight")

st.write("""
- Transaksi dengan probabilitas tinggi perlu investigasi lebih lanjut
- Model membantu mendeteksi pola fraud secara otomatis
- Threshold dapat diatur sesuai toleransi risiko bisnis
""")

# =========================
# DOWNLOAD
# =========================
st.subheader("⬇️ Download Hasil")

csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download Result CSV", csv, "fraud_result.csv", "text/csv")