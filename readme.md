# Walmart Hackathon Prototype 🛒

A Flask-based web app that predicts optimal disposition for returned products using machine learning. Built for a Walmart-focused hackathon challenge.

---

## 🚀 Overview

- Web interface enables users to input product name, original price, refund reason, and comments.
- ML models provide:
  - Estimated **refurbishment cost**
  - Expected **resale value**
  - **Profit calculation**
  - **Recommended decision** (Refurbish, Liquidate, Recycle, Donate, Keep It)
- Four-page flow with dynamic user feedback (`detail`, `reason`, `payment`, `success`, plus a dashboard).
- Decision prediction powered by **three pickled models**.

---

## ⚙️ Features

- **Modular workflow** using session-like storage (`last_prediction`).
- Predictive logic based on user input and engineered features like `refurb_ratio`, `resale_ratio`, `profit`, etc.
- Clear mapping of return reasons and product categories to model codes.
- Flexible to extend more products, reasons, and logic.

---

## 📁 File Structure

``
├── app.py
├── models/
│ ├── model_refurb.pkl
│ ├── model_resale.pkl
│ └── model_decision.pkl
├── templates/
│ ├── walmart.html
│ ├── detail1.html
│ ├── reason2.html
│ ├── payment3.html
│ ├── success_page4.html
│ └── dashboard.html
├── README.md
└── requirements.txt


## pip install -r requirements.txt

numpy
pandas 
scikit-learn 
lightgbm 
matplotlib

streamlit
flask
pickle
lightgbm 
groq




## Create & activate a virtual env

python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

## predict example

{
  "predicted_refurb_cost": 120.00,
  "predicted_resale_value": 250.00,
  "profit": 50.00,
  "recommended_decision": "Refurbish",
  "action_explanation": "The model suggests this action based on predicted profitability."
}
