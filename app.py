import streamlit as st
import joblib
import numpy as np

# Load the scaler and models
scaler = joblib.load("models/scaler.joblib")
lr_model = joblib.load("models/linear_regression_model.joblib")
dt_model = joblib.load("models/decision_tree_model.joblib")

# Title of the app
st.title("Sierra Leone House Price Prediction")

# Input fields for user
median_income = st.number_input("Median Income (USD)", min_value=0, step=100)
population_density = st.number_input("Population Density (people/sq km)", min_value=0, step=10)
rooms_per_house = st.number_input("Average Rooms per House", min_value=1, step=1)

# Predict button
if st.button("Predict"):
    # Create a feature array
    features = np.array([[median_income, population_density, rooms_per_house]])
    
    # Scale the features
    scaled_features = scaler.transform(features)
    
    # Make predictions
    lr_prediction = lr_model.predict(scaled_features)[0]
    dt_prediction = dt_model.predict(scaled_features)[0]
    
    # Display predictions
    st.subheader("Predicted House Prices:")
    st.write(f"Linear Regression: ${lr_prediction:.2f}")
    st.write(f"Decision Tree: ${dt_prediction:.2f}")