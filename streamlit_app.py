import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime

# Load model and scaler
model = joblib.load("water_quality_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define features
features = ['NH4', 'BSK5', 'Suspended', 'O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

# UI Setup
st.set_page_config(page_title="Water Quality Predictor", page_icon="💧", layout="centered")
st.title("💧 Water Quality Predictor")
st.markdown("Enter water test parameters to predict potability.")

# Sidebar info
st.sidebar.title("🔧 Model Info")
st.sidebar.markdown("**Model:** Random Forest Classifier")
st.sidebar.markdown("**Trained on:** PB_All_2000_2021.csv")
st.sidebar.markdown("**Accuracy:** 91.2%")
st.sidebar.markdown("**Date:** " + str(datetime.date.today()))

# User inputs
user_input = []
for feat in features:
    val = st.number_input(f"{feat}:", step=0.1, format="%.2f")
    user_input.append(val)

# Explanation section
with st.expander("ℹ️ About the Features"):
    st.markdown("""
    - **NH4**: Ammonium (mg/L)
    - **BSK5**: Biochemical Oxygen Demand (mg/L)
    - **Suspended**: Suspended Solids (mg/L)
    - **O2**: Dissolved Oxygen (mg/L)
    - **NO3**: Nitrate (mg/L)
    - **NO2**: Nitrite (mg/L)
    - **SO4**: Sulfate (mg/L)
    - **PO4**: Phosphate (mg/L)
    - **CL**: Chloride (mg/L)
    """)

# Predict Button
if st.button("🔍 Predict"):
    input_scaled = scaler.transform([user_input])
    result = model.predict(input_scaled)[0]

    st.subheader("📊 Prediction Result:")
    if result == 1:
        st.success("✅ Water is Safe to Drink")
        st.balloons()
        report_text = "✅ Water is safe to drink.\n\nKeep monitoring regularly!"
    else:
        st.error("❌ Water is NOT Safe to Drink")
        st.warning("⚠️ Consider treatment for pollutants.")
        report_text = "❌ Water is NOT safe.\n\nTake appropriate water treatment measures."

    # 📈 Radar Chart using go.Figure (compatible with pandas >= 2.0)
    radar_df = pd.DataFrame({
        'Feature': features,
        'Value': user_input
    })

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=radar_df['Value'],
        theta=radar_df['Feature'],
        fill='toself',
        name='Input Values'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True),
        ),
        showlegend=False,
        title="📈 Feature Radar Chart"
    )

    st.plotly_chart(fig)

    # 📄 Download Report
    st.download_button(
        label="📄 Download Report",
        data=report_text,
        file_name=f"water_quality_report_{datetime.date.today()}.txt",
        mime='text/plain'
    )

# 📊 Optional: Compare with dataset average
with st.expander("📊 Compare with Average from Dataset"):
    try:
        data = pd.read_csv("PB_All_2000_2021.csv")
        avg = data[features].mean().round(2).to_dict()
        st.write("🧪 Average Values in Dataset:")
        st.json(avg)
    except:
        st.warning("⚠️ Dataset file not found on Desktop.")

