import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Latin America Regression Explorer", layout="wide")
st.title("📊 Latin America Regression Explorer")
st.write("**By Racely Ortega**")

# --------------------------
# Sample Data
# --------------------------
years = np.arange(1960, 2021, 5)
categories = ["Population", "Unemployment rate", "Education levels", "Life expectancy",
              "Average wealth", "Average income", "Birth rate", "Immigration out", "Murder Rate"]

# Minimal example data for all countries
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
    }),
    # You can continue adding all other Latin countries, filling with placeholder/sparse data if needed
}

# --------------------------
# User selection
# --------------------------
category = st.selectbox("Select a category:", categories)
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
    df = data_samples[c]
    if category not in df.columns:
        st.warning(f"No data for {category} in {c}.")
    else:
        st.data_editor(df[["Year", category]], key=f"data_{idx}_{c}")

# --------------------------
# Global year range
# --------------------------
all_years = np.concatenate([data_samples[c]["Year"].values for c in countries])
min_year = all_years.min()
max_year = all_years.max()
years_plot = np.arange(min_year, max_year + extrapolate_years + 1, increment)

# --------------------------
# Regression, Plot, Analysis
# --------------------------
st.subheader("📈 Regression, Function Analysis & Predictions")
fig, ax = plt.subplots(figsize=(12,6))
analysis_results = {}

for c in countries:
    df = data_samples[c]
    if category not in df.columns:
        continue
    X = df["Year"].values.reshape(-1,1)
    y = df[category].values
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)

    # Regression curve
    X_plot = poly.transform(years_plot.reshape(-1,1))
    y_plot = model.predict(X_plot)
    ax.scatter(X.flatten(), y, label=f"{c} data")
    ax.plot(years_plot, y_plot, label=f"{c} regression")

    # Regression equation
    coefs = model.coef_
    intercept = model.intercept_
    terms = [f"{round(coefs[i],2)}*x^{i}" for i in range(len(coefs))]
    equation = " + ".join(terms) + f" + {round(intercept,2)}"
    st.markdown(f"**{c} Regression Equation:** {equation}")

    # Function analysis
    dy = np.gradient(y_plot, years_plot)
    max_idx = np.argmax(y_plot)
    min_idx = np.argmin(y_plot)
    fast_inc = np.argmax(dy)
    fast_dec = np.argmin(dy)
    domain = (years_plot.min(), years_plot.max())
    range_vals = (y_plot.min(), y_plot.max())

    st.markdown(f"**Function Analysis for {c}:**")
    st.write(f"- Local maximum: {y_plot[max_idx]:.2f} in {years_plot[max_idx]}")
    st.write(f"- Local minimum: {y_plot[min_idx]:.2f} in {years_plot[min_idx]}")
    st.write(f"- Increasing years: {years_plot[dy>0][0]} to {years_plot[dy>0][-1]}")
    st.write(f"- Decreasing years: {years_plot[dy<0][0]} to {years_plot[dy<0][-1]}")
    st.write(f"- Fastest increase: {dy[fast_inc]:.2f} per year in {years_plot[fast_inc]}")
    st.write(f"- Fastest decrease: {dy[fast_dec]:.2f} per year in {years_plot[fast_dec]}")
    st.write(f"- Domain: {domain}")
    st.write(f"- Range: {range_vals}")
    st.write(f"- Conjecture: Significant changes may be due to economic or social shifts in {c}.")

    analysis_results[c] = {"model":model, "poly":poly, "years":years_plot, "y_pred":y_plot, "X":X, "y":y}

ax.set_xlabel("Year")
ax.set_ylabel(category)
ax.legend()
st.pyplot(fig)

# --------------------------
# Predictions / Interpolation / Extrapolation
# --------------------------
st.subheader("🔮 Predictions")
pred_year = st.number_input("Enter a year to predict:", min_value=1950, max_value=2100, value=2030)
for c, res in analysis_results.items():
    pred_val = res["model"].predict(res["poly"].transform([[pred_year]]))[0]
    st.write(f"In {pred_year}, predicted {category} for {c}: {pred_val:.2f}")

# --------------------------
# Average Rate of Change
# --------------------------
st.subheader("📐 Average Rate of Change")
y1 = st.number_input("Start year:", min_value=int(min_year), max_value=int(max_year), value=int(min_year))
y2 = st.number_input("End year:", min_value=int(min_year), max_value=int(max_year+extrapolate_years), value=int(max_year))
if y2 > y1:
    for c, res in analysis_results.items():
        val1 = res["model"].predict(res["poly"].transform([[y1]]))[0]
        val2 = res["model"].predict(res["poly"].transform([[y2]]))[0]
        avg_rate = (val2 - val1)/(y2 - y1)
        st.write(f"Avg rate of change for {c} between {y1}-{y2}: {avg_rate:.2f} units/year")

# --------------------------
# US Latin Groups Comparison (illustrative)
# --------------------------
st.subheader("🇺🇸 Latin Groups in the US")
us_groups = {
    "Mexican-Americans": np.random.randint(50, 90, len(years)),
    "Puerto Ricans": np.random.randint(55, 85, len(years)),
    "Cuban-Americans": np.random.randint(60, 95, len(years)),
}
compare_us = st.checkbox("Show comparison with US Latin groups")
if compare_us:
    fig2, ax2 = plt.subplots(figsize=(12,5))
    for g, vals in us_groups.items():
        ax2.plot(years, vals, label=g)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Index Value")
    ax2.legend()
    st.pyplot(fig2)

# --------------------------
# Printer-Friendly Report
# --------------------------
st.subheader("🖨️ Printer-Friendly Report")
report_text = "Latin America Regression Analysis\nBy Racely Ortega\n\n"
for c, res in analysis_results.items():
    report_text += f"{c}:\n"
    report_text += f"Equation: {res['model'].coef_} \n"
report_file = st.text_area("Report Preview", value=report_text, height=200)
st.download_button("Download Report", data=report_text, file_name="report.txt")
