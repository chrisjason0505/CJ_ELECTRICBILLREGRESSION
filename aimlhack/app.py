import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Set Streamlit page config
st.set_page_config(page_title="Electricity Bill Predictor", layout="wide")

# Load model
model = joblib.load("electricity_model.pkl")

# Sample correlation data (you can replace this with actual df.corr() from training set)
# These are placeholder values – update with your actual correlations if needed

import pandas as pd

# Your correlation dictionary
correlation_data = {
    'Fan': 0.410682,
    'Refrigerator': 0.376816,
    'AirConditioner':0.261845,
    'Television': 0.412651,
    'Monitor': 0.309986,
    'MonthlyHours':0.958702,
    'TariffRate': 0.28622,
    'Month':0.036316,
    'motor_pump':0
}

# Convert to DataFrame and sort
corr_df = pd.DataFrame({
    'Feature': list(correlation_data.keys()),
    'Correlation with Bill': list(correlation_data.values())
}).sort_values('Correlation with Bill', ascending=False)

# Plot
# Plot
fig = px.bar(
    corr_df,
    x='Feature',
    y='Correlation with Bill',
    title='Feature Correlation with Electricity Bill',
    text='Correlation with Bill',
    labels={'Correlation with Bill': 'Correlation'},
    color='Correlation with Bill',
    color_continuous_scale='Blues'
)
fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig.update_layout(yaxis=dict(range=[0, 1]), uniformtext_minsize=8)

# ✅ Show it in Streamlit
st.plotly_chart(fig, use_container_width=True)


# Sidebar layout
with st.sidebar:
    st.title("⚡ Bill Prediction Inputs")

    fan = st.number_input("Fan Usage (kWh)", min_value=0, step=1)
    fridge = st.number_input("Refrigerator Usage (kWh)", min_value=0, step=1)
    ac = st.number_input("Air Conditioner Usage (kWh)", min_value=0, step=1)
    tv = st.number_input("Television Usage (kWh)", min_value=0, step=1)
    monitor = st.number_input("Monitor Usage (kWh)", min_value=0, step=1)
    monthly_hours = st.slider("Monthly Appliance Usage (Hours)", min_value=100,max_value=1000, step=1)
    tariff_rate = st.slider("Tariff Rate (₹ per unit)", min_value=5.0, max_value=15.0, step=0.1, value=8.0)

# Fixed value for MotorPump
motor_pump = 0 #corr is Nan

# Input array in correct order
input_data = np.array([[fan, fridge, ac, tv, monitor, motor_pump, monthly_hours, tariff_rate]])

# Main layout
st.title("Electricity Bill Prediction Dashboard")
st.markdown("This app predicts your estimated monthly electricity bill based on parameters like appliance usage and tariff rate.")

if st.button("🔍 Predict Electricity Bill"):
    prediction = model.predict(input_data)
    st.success(f"💡 Estimated Monthly Electricity Bill: ₹{prediction[0]:,.2f}")

