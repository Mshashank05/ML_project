import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('car_sales_prediction_model.pkl')

# Streamlit app title
st.title("Car Sales Prediction App")

# Input fields for user to enter car details
price = st.number_input('Price in thousands', min_value=0.0, value=25.0)
engine_size = st.number_input('Engine Size (L)', min_value=0.0, value=2.0)
horsepower = st.number_input('Horsepower', min_value=0.0, value=150.0)
fuel_efficiency = st.number_input('Fuel Efficiency (mpg)', min_value=0.0, value=25.0)

# Prediction button
if st.button('Predict Sales'):
    # Prepare features for prediction (without scaling)
    features = np.array([[price, engine_size, horsepower, fuel_efficiency]])

    # Make prediction using the loaded model
    prediction = model.predict(features)

    # Display the predicted sales
    st.success(f"Predicted Sales: {prediction[0]:.2f} thousand units")
