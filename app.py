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
# SAMPLE DATA
# --------------------------
data_samples = {
    "Brazil": pd.DataFrame({"Year": np.arange(1960,2021,5),"Population":[720,770,820,880,940,1000,1060,1130,1200,1270,1350,1430,1510]}),
    "Mexico": pd.DataFrame({"Year": np.arange(1960,2021,5),"Population":[380,410,440,470,500,530,560,590,620,650,680,710,740]}),
    "Argentina": pd.DataFrame({"Year": np.arange(1960,2021,5),"Population":[20,22,24,26,28,30,32,34,36,38,40,42,44]}),
    "Colombia": pd.DataFrame({"Year": np.arange(1960,2021,5),"Population":[20,23,26,29,32,35,38,41,44,47,50,53,56]}),
    "Chile": pd.DataFrame({"Year": np.arange(1960,2021,5),"Population":[8,9,10,11,12,13,14,15,16,17,18,19,20]}),
    "Peru": pd.DataFrame({"Year": np.arange(1960,2021,5),"Population":[10,11,12,13,14,15,16,17,18,19,20,21,22]}),
    "Venezuela": pd.DataFrame({"Year": np.arange(1960,2021,5),"Population":[8,9,10,11,12,13,14,15,16,17,18,19,20]}),
    "Ecuador": pd.DataFrame({"Year": np.arange(1960,2021,5),"Population":[5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11]}),
    "Guatemala": pd.DataFrame({"Year": np.arange(1960,2021,5),"Population":[3,3.2,3.5,3.8,4,4.3,4.6,4.9,5.2,5.5,5.8,6.1,6.5]}),
    "Dominican Republic": pd.DataFrame({"Year": np.arange(1960,2021,5),"Population":[2,2.2,2.5,2.8,3,3.3,3.6,3.9,4.2,4.5,4.8,5.1,5.5]})
}

categories = ["Population"]

category = st.selectbox("Select a data category:", categories)
degree = st.slider("Select polynomial regression degree:", 3, 8, 3)
increment = st.slider("Graph increments (years):", 1, 10, 1)
extrapolate_years = st.slider("Extrapolate into future (years):", 0, 20, 5)
countries = st.multiselect("Select countries to analyze:", list(data_samples.keys()), default=list(data_samples.keys()))

# --------------------------
# Raw Data
# --------------------------
st.subheader("📋 Raw Data (Editable Tables)")
for idx, c in enumerate(countries):
    st.write(f"### {c}")
    st.data_editor(data_samples[c][["Year", category]], num_rows="dynamic", key=f"data_editor_{idx}_{c}")

# --------------------------
# Regression & Plot
# --------------------------
st.subheader("📈 Polynomial Regression Plot")
fig, ax = plt.subplots(figsize=(10,6))
analysis_results = {}
for c in countries:
    df = data_samples[c]
    X = df["Year"].values.reshape(-1,1)
    y = df[category].values
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    
    # Years for plotting
    years = np.arange(df["Year"].min(), df["Year"].max()+extrapolate_years+1, increment)
    X_plot = poly.transform(years.reshape(-1,1))
    y_plot = model.predict(X_plot)
    
    # Original data
    ax.scatter(df["Year"], y, label=f"{c} data")
    
    # Regression curve
    ax.plot(years, y_plot, label=f"{c} regression")
    
    # Regression equation
    coefs = model.coef_
    intercept = model.intercept_
    terms = [f"{round(coefs[i],2)}*x^{i}" for i in range(len(coefs))]
    equation = " + ".join(terms) + f" + {round(intercept,2)}"
    st.markdown(f"**{c} Regression Equation:** {equation}")
    
    # Store for function analysis
    analysis_results[c] = {"model":model, "poly":poly, "years":years}

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
    years = analysis_results[c]["years"]
    X_pred = poly.transform(years.reshape(-1,1))
    y_pred = model.predict(X_pred)
    dy = np.gradient(y_pred, years)
    ddy = np.gradient(dy, years)
    
    max_idx = np.argmax(y_pred)
    min_idx = np.argmin(y_pred)
    max_growth_idx = np.argmax(dy)
    max_decline_idx = np.argmin(dy)
    
    st.write(f"### {c}")
    st.write(f"- Local maximum: {round(y_pred[max_idx],2)} at {years[max_idx]}")
    st.write(f"- Local minimum: {round(y_pred[min_idx],2)} at {years[min_idx]}")
    st.write(f"- Increasing years: {years[dy>0][0]} to {years[dy>0][-1]}")
    st.write(f"- Decreasing years: {years[dy<0][0]} to {years[dy<0][-1]}")
    st.write(f"- Fastest growth: {round(dy[max_growth_idx],2)} units/year at {years[max_growth_idx]}")
    st.write(f"- Fastest decline: {round(dy[max_decline_idx],2)} units/year at {years[max_decline_idx]}")
    st.write(f"- Domain: {years[0]} to {years[-1]}")
    st.write(f"- Range: {round(min(y_pred),2)} to {round(max(y_pred),2)}")

# --------------------------
# Prediction / Interpolation / Extrapolation
# --------------------------
st.subheader("🔮 Prediction")
pred_year = st.number_input("Enter a year to predict:", min_value=1950, max_value=2100, value=2030)
for c in countries:
    model = analysis_results[c]["model"]
    poly = analysis_results[c]["poly"]
    pred_val = model.predict(poly.transform([[pred_year]]))[0]
    st.write(f"{c} predicted {category} in {pred_year}: {round(pred_val,2)}")

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
        avg_rate = (val2-val1)/(y2-y1)
        st.write(f"{c} average rate of change between {y1}-{y2}: {round(avg_rate,2)} units/year")

# --------------------------
# US Latin Group Comparison (illustrative)
# --------------------------
st.subheader("🇺🇸 US Latin Groups (Illustrative)")
us_groups = {
    "Mexican-Americans": np.random.randint(50,90, len(np.arange(1960,2021,5))),
    "Puerto Ricans": np.random.randint(55,85, len(np.arange(1960,2021,5))),
    "Cuban-Americans": np.random.randint(60,95, len(np.arange(1960,2021,5)))
}
show_us = st.checkbox("Show comparison with US Latin groups")
if show_us:
    fig2, ax2 = plt.subplots()
    years = np.arange(1960,2021,5)
    for g,v in us_groups.items():
        ax2.plot(years,v,label=g)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Index Value")
    ax2.legend()
    st.pyplot(fig2)

# --------------------------
# Printer-Friendly Report
# --------------------------
st.subheader("🖨️ Printer-Friendly Report")
st.download_button(
    "Download Report",
    "Full report with data, regression analysis, function analysis, predictions, charts.\nBy Racely Ortega",
    file_name="report.txt")
