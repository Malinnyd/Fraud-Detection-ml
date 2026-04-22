# 🚨 Fraud Detection System (Machine Learning)

A machine learning-based fraud detection system using **XGBoost** and **Streamlit** to identify potentially fraudulent financial transactions in real-time.

---

## 📌 Project Overview
This project aims to detect fraudulent transactions using supervised machine learning. The model is trained on anonymized credit card transaction data and deployed as an interactive web application using Streamlit.

---

## ⚙️ Features
- 📊 Upload custom transaction dataset (CSV)
- 🤖 Fraud prediction using trained XGBoost model
- 📈 Fraud probability scoring
- 🔥 Top high-risk transaction detection
- 📉 Visualization of fraud probability distribution
- ⬇️ Download prediction results

---

## 🧠 Machine Learning Model
- Algorithm: **XGBoost Classifier**
- Technique: **SMOTE (handling class imbalance)**
- Target: Fraud detection (binary classification)

---

## 🗂️ Dataset
- Dataset: Credit Card Fraud Detection
- Features:
  - `V1 - V28` (PCA transformed features)
  - `Amount`
  - `Time`
- Target:
  - `Class` (0 = Normal, 1 = Fraud)

---

## 🔧 Installation & Run Locally

```bash
git clone https://github.com/USERNAME/fraud-detection-ml.git
cd fraud-detection-ml
pip install -r requirements.txt
streamlit run app.py
