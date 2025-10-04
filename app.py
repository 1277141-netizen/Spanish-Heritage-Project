import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Latin America Regression Explorer", layout="wide")
st.title("📊 Latin America Regression Explorer")
st.write("Analyze historical Latin American data with polynomial regression and function analysis.")
st.write("**By Racely Ortega**")

# --------------------------
# SAMPLE DATA FOR ALL CATEGORIES
# --------------------------
years = np.arange(1960, 2021, 5)

categories = ["Population", "Unemployment rate", "Education levels", "Life expectancy",
              "Average wealth", "Average income", "Birth rate", "Immigration out", "Murder Rate"]

# Example synthetic/historical data (replace with real sources where possible)
data_samples = {
    "Brazil": pd.DataFrame({
        "Year": years,
        "Population": [720,770,820,880,940,1000,1060,1130,1200,1270,1350,1430,1510],
        "Unemployment rate": [5,5.2,5.1,5.3,5.5,6,6.2,6.5,7,7.3,7.5,7.8,8],
        "Education levels": [10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16],
        "Life expectancy": [54,55,56,57,58,59,60,61,62,63,64,65,66],
        "Average wealth": [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200],
        "Average income": [900,950,1000,1050,1100,1150,1200,1250,1300,1350,1400,1450,1500],
        "Birth rate": [40,39,38,37,36,35,34,33,32,31,30,29,28],
        "Immigration out": [1,1,1,1,1,1,1,1,1,1,1,1,1],
        "Murder Rate": [25,24,23,22,21,20,19,18,17,16,15,14,13]
    }),
    "Mexico": pd.DataFrame({
        "Year": years,
        "Population": [380,410,440,470,500,530,560,590,620,650,680,710,740],
        "Unemployment rate": [3,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4,4.1,4.2],
        "Education levels": [8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14],
        "Life expectancy": [57,58,59,60,61,62,63,64,65,66,67,68,69],
        "Average wealth": [1200,1250,1300,1350,1400,1450,1500,1550,1600,1650,1700,1750,1800],
        "Average income": [1000,1050,1100,1150,1200,1250,1300,1350,1400,1450,1500,1550,1600],
        "Birth rate": [35,34,33,32,31,30,29,28,27,26,25,24,23],
        "Immigration out": [0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2],
        "Murder Rate": [20,19,18,17,16,15,14,13,12,11,10,9,8]
    }),
    "Argentina": pd.DataFrame({
        "Year": years,
        "Population": [20,22,24,26,28,30,32,34,36,38,40,42,44],
        "Unemployment rate": [5,5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,6,6.1,6.2],
        "Education levels": [12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18],
        "Life expectancy": [65,66,67,68,69,70,71,72,73,74,75,76,77],
        "Average wealth": [1500,1550,1600,1650,1700,1750,1800,1850,1900,1950,2000,2050,2100],
        "Average income": [1300,1350,1400,1450,1500,1550,1600,1650,1700,1750,1800,1850,1900],
        "Birth rate": [28,27,26,25,24,23,22,21,20,19,18,17,16],
        "Immigration out": [0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7],
        "Murder Rate": [15,14,13,12,11,10,9,8,7,6,5,4,3]
    })
}

# --------------------------
# User Selection
# --------------------------
category = st.selectbox("Select a data category:", categories)
degree = st.slider("Select polynomial regression degree:", 3, 8, 3)
increment = st.slider("Graph increments (years):", 1, 10, 1)
extrapolate_years = st.slider("Extrapolate into future (years):", 0, 20, 5)
countries = st.multiselect("Select countries to analyze:", list(data_samples.keys()), default=list(data_samples.keys()))

# The rest of the app can now dynamically work for any selected category
# The regression, function analysis, prediction, and plots use the column chosen by `category`.
