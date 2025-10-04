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
# Sample Data (replace with real data if available)
# --------------------------
years = np.arange(1960, 2021, 5)
categories = ["Population", "Unemployment rate", "Education levels", "Life expectancy",
              "Average wealth", "Average income", "Birth rate", "Immigration out", "Murder Rate"]

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
degree = st.slider("Polynomial degree:", 3, 8, 3)
increment = st.slider("Graph increments (years):", 1, 10, 1)
extrapolate_years = st.slider("Extrapolate future years:", 0, 20, 5)
countries = st.multiselect("Select countries:", list(data_samples.keys()), default=list(data_samples.keys()))

# --------------------------
# Raw Data Tables
# --------------------------
st.subheader("📋 Raw Data")
for idx, c in enumerate(countries):
    st.write(f"### {c}")
    st.data_editor(data_samples[c][["Year", category]], key=f"data_{idx}_{c}")

# --------------------------
# Regression & Plots
# --------------------------
st.subheader("📈 Regression Plot")
fig, ax = plt.subplots(figsize=(10,6))
analysis_results = {}
for c in countries:
    df = data_samples[c]
    X = df["Year"].values.reshape(-1,1)
    y = df[category].values
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    
    years_plot = np.arange(df["Year"].min(), df["Year"].max()+extrapolate_years+1, increment)
    X_plot = poly.transform(years_plot.reshape(-1,1))
    y_plot = model.predict(X_plot)
    
    ax.scatter(df["Year"], y, label=f"{c} data")
    ax.plot(years_plot, y_plot, label=f"{c} regression")
    
    coefs = model.coef_
    intercept = model.intercept_
    terms = [f"{round(coefs[i],2)}*x^{i}" for i in range(len(coefs))]
    equation = " + ".join(terms) + f" + {round(intercept,2)}"
    st.markdown(f"**{c} Regression Equation:** {equation}")
    
    analysis_results[c] = {"model":model, "poly":poly, "years":years_plot}

ax.set_xlabel("Year")
ax.set_ylabel(category)
ax.legend()
st.pyplot(fig)

# --------------------------
# Function Analysis
# --------------------------
st.subheader("🔍 Function Analysis")
for c in countries:
    model = analysis_results[c]["model"]
    poly = analysis_results[c]["poly"]
    years_arr = analysis_results[c]["years"]
    X_pred = poly.transform(years_arr.reshape(-1,1))
    y_pred = model.predict(X_pred)
    dy = np.gradient(y_pred, years_arr)
    
    max_idx = np.argmax(y_pred)
    min_idx = np.argmin(y_pred)
    max_growth_idx = np.argmax(dy)
    max_decline_idx = np.argmin(dy)
    
    inc_years = years_arr[dy>0]
    dec_years = years_arr[dy<0]
    
    st.write(f"### {c}")
    st.write(f"- Local maximum: {round(y_pred[max_idx],2)} at {years_arr[max_idx]}")
    st.write(f"- Local minimum: {round(y_pred[min_idx],2)} at {years_arr[min_idx]}")
    st.write(f"- Increasing years: {inc_years[0]} to {inc_years[-1]}" if len(inc_years)>0 else "- Increasing years: None")
    st.write(f"- Decreasing years: {dec_years[0]} to {dec_years[-1]}" if len(dec_years)>0 else "- Decreasing years: None")
    st.write(f"- Fastest growth: {round(dy[max_growth_idx],2)} units/year at {years_arr[max_growth_idx]}")
    st.write(f"- Fastest decline: {round(dy[max_decline_idx],2)} units/year at {years_arr[max_decline_idx]}")
    st.write(f"- Domain: {years_arr[0]} to {years_arr[-1]}")
    st.write(f"- Range: {round(min(y_pred),2)} to {round(max(y_pred),2)}")
    st.write(f"- Conjecture: Significant changes may relate to economic or social events affecting {c} during the period.\n")
    
# --------------------------
# Prediction / Interpolation / Extrapolation
# --------------------------
st.subheader("🔮 Predict Year Value")
pred_year = st.number_input("Enter year to predict:", min_value=1950, max_value=2100, value=2030)
for c in countries:
    model = analysis_results[c]["model"]
    poly = analysis_results[c]["poly"]
    pred_val = model.predict(poly.transform([[pred_year]]))[0]
    st.write(f"In {pred_year}, predicted {category} for {c}: {round(pred_val,2)} units")

# --------------------------
# Average Rate of Change
# --------------------------
st.subheader("📐 Average Rate of Change")
y1 = st.number_input("Start year:", min_value=1950, max_value=2100, value=1960, key="start")
y2 = st.number_input("End year:", min_value=1950, max_value=2100, value=2020, key="end")
if y2>y1:
    for c in countries:
        model = analysis_results[c]["model"]
        poly = analysis_results[c]["poly"]
        val1 = model.predict(poly.transform([[y1]]))[0]
        val2 = model.predict(poly.transform([[y2]]))[0]
        avg_rate = (val2 - val1)/(y2 - y1)
        st.write(f"Avg rate of change for {c} between {y1}-{y2}: {round(avg_rate,2)} units/year")

# --------------------------
# US Latin Groups Comparison
# --------------------------
st.subheader("🇺🇸 US Latin Groups Comparison (Illustrative)")
us_groups = {
    "Mexican-Americans": np.random.randint(50,90,len(years)),
    "Puerto Ricans": np.random.randint(55,85,len(years)),
    "Cuban-Americans": np.random.randint(60,95,len(years)),
}
fig2, ax2 = plt.subplots(figsize=(10,5))
for g,v in us_groups.items():
    ax2.plot(years, v, label=g)
ax2.set_xlabel("Year")
ax2.set_ylabel("Index Value")
ax2.legend()
st.pyplot(fig2)

# --------------------------
# Printer-Friendly Report
# --------------------------
st.subheader("🖨️ Printer-Friendly Report")
report_text = "Regression analysis, function analysis, predictions, and plots.\nBy Racely Ortega"
st.download_button("Download Report", report_text, file_name="report.txt")
