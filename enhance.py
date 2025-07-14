import streamlit as st
import pickle
import pandas as pd

# Load your trained model
model = pickle.load(open("return_mode_model.pkl", "rb"))

st.title("Return-Right Scorer Walmart")

# User input fields
price = st.number_input("Enter Price", min_value=0.0, step=1.0)
resale_value = st.number_input("Enter Resale Value", min_value=0.0, step=1.0)


if st.button("Predict"):
    # Prepare input as DataFrame
    input_data = pd.DataFrame(
        [[resale_value, price]],
        columns=["resale_value", "price"]
    )
    
    # Predict using model
    pred = model.predict(input_data)
    
    # Show result
    st.success(f"Prediction = :: {pred[0]}")
