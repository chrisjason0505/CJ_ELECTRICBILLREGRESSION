import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import os

# 🚧 Load model safely using absolute path
model_path = os.path.join(os.path.dirname(__file__), "electricity_model.pkl")
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("❌ Model file 'electricity_model.pkl' not found. Please upload it to the same directory.")
    st.stop()

# ⚙️ Page config
st.set_page_config(page_title="Electricity Bill Predictor", layout="wide")

# 📈 Correlation Data (replace with real data if needed)
correlation_data = {
    'Fan': 0.410682,
    'Refrigerator': 0.376816,
    'AirConditioner': 0.261845,
    'Television': 0.412651,
    'Monitor': 0.309986,
    'MonthlyHours': 0.958702,
    'TariffRate': 0.28622,
    'Month': 0.036316,
    'motor_pump': 0
}

corr_df = pd.DataFrame({
    'Feature': list(correlation_data.keys()),
    'Correlation with Bill': list(correlation_data.values())
}).sort_values('Correlation with Bill', ascending=False)

# 📊 Correlation Plot
fig = px.bar(
    corr_df,
    x='Feature',
    y='Correlation with Bill',
    title='📊 Feature Correlation with Electricity Bill',
    text='Correlation with Bill',
    labels={'Correlation with Bill': 'Correlation'},
    color='Correlation with Bill',
    color_continuous_scale='Blues'
)
fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig.update_layout(yaxis=dict(range=[0, 1]), uniformtext_minsize=8)

st.plotly_chart(fig, use_container_width=True)

# 🧾 Sidebar for input
with st.sidebar:
    st.title("⚡ Bill Prediction Inputs")
    fan = st.number_input("Fan Usage (kWh)", min_value=0, step=1)
    fridge = st.number_input("Refrigerator Usage (kWh)", min_value=0, step=1)
    ac = st.number_input("Air Conditioner Usage (kWh)", min_value=0, step=1)
    tv = st.number_input("Television Usage (kWh)", min_value=0, step=1)
    monitor = st.number_input("Monitor Usage (kWh)", min_value=0, step=1)
    monthly_hours = st.slider("Monthly Appliance Usage (Hours)", min_value=100, max_value=1000, step=1)
    tariff_rate = st.slider("Tariff Rate (₹ per unit)", min_value=5.0, max_value=15.0, step=0.1, value=8.0)

# 🧠 Build input in model order
motor_pump = 0  # You set this fixed; adjust as needed
input_data = np.array([[fan, fridge, ac, tv, monitor, motor_pump, monthly_hours, tariff_rate]])

# 📍 Main area
st.title("💡 Electricity Bill Prediction Dashboard")
st.markdown("Predict your estimated monthly electricity bill based on appliance usage and tariff rate.")

# 🔍 Predict
if st.button("🔍 Predict Electricity Bill"):
    try:
        prediction = model.predict(input_data)
        st.success(f"💰 Estimated Monthly Electricity Bill: ₹{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
