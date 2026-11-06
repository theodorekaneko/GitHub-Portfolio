import streamlit as st
import pandas as pd
from tweedie_regressor import predict_loss_cost

st.set_page_config(
    page_title='Loss Cost Predictor', 
    layout='centered'
    )

st.title('Welcome to the Loss Cost Predictor')


VehPow = st.slider("Classify vehicle power", min_value=4, max_value=15, value=6)
VehAge = st.slider("What is the age of your vehicle?", min_value=0, max_value=100, value=6)
DrivAge = st.slider("What is your age", min_value=18, max_value=100, value=44)
BonusMalus = st.slider("What is your bonus/malus rating", min_value=50, max_value=230, value=50)
VehBrand = st.radio("Select vehicle brand:", options=['B12', 'B1', 'B2', 'B3', 'B5'], index=None)
VehGas = st.radio("Select vehicle gas type:", options=['Regular', 'Diesel'], index=None)
Area = st.radio("Select area('A' - rural to 'F' - urban centre)", options=['A', 'B', 'C', 'D', 'E', 'F'], index=None)
Density = st.slider("What is the density of your area?", min_value=1, max_value=27000, value=393)
Region = st.radio("Select Region:", options=['Centre', 'Rhone-Alpes', "Provence-Alpes-Cotes-D'Azur", 'Ile-de-France', 'Bretagne'], index=None)
Exposure = st.slider("Enter Exposure: ", min_value=0.0, max_value=2.01, value=0.49)

if st.button("Predict Pure Premium"):
    final_predicted_rate = predict_loss_cost(VehPow, VehAge, DrivAge, BonusMalus, VehBrand, VehGas, Area, Density, Region, Exposure)
    st.write(f"\nPredicted Pure Premium: \n${final_predicted_rate:,.2f}")