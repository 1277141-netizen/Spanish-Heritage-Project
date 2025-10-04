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
# SAMPLE DATA (built-in, replaceable)
# --------------------------
data_samples = {
    "Brazil": pd.DataFrame({
        "Year": np.arange(1960, 2021, 5),
        "Population": [720, 770, 820, 880, 940, 1000, 1060, 1130, 1200, 1270, 1350, 1430, 1510],
        "GDP": [500, 550, 600, 670, 740, 810, 900, 1000, 1100, 1200, 1350, 1500, 1700]
    }),
    "Mexico": pd.DataFrame({
        "Year": np.arange(1960, 2021, 5),
        "Population": [380, 410, 440, 470, 500, 530, 560, 590, 620, 650, 680, 710, 740],
        "GDP": [300, 320, 350, 390, 430, 480, 540, 600, 670, 740, 820, 910, 1000]
    }),
    "Argentina": pd.DataFrame({
        "Year": np.arange(1960, 2021, 5),
        "Population": [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44],
        "GDP": [200, 220, 240, 260, 280, 300, 330, 360, 390, 420, 450, 480, 510]
    }),
    "Colombia": pd.DataFrame({
        "Year": np.arange(1960, 2021, 5),
        "Population": [20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56],
        "GDP": [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 330, 360]
    }),
    "Chile": pd.DataFrame({
        "Year": np.arange(1960, 2021, 5),
        "Population": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "GDP": [50, 55, 60, 66, 72, 78, 85, 92, 100, 110, 120, 130, 140]
    }),
    "Peru": pd.DataFrame({
        "Year": np.arange(1960, 2021, 5),
        "Population": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        "GDP": [40, 45, 50, 55, 60, 65, 70, 76, 82, 88, 95, 102, 110]
    }),
    "Venezuela": pd.DataFrame({
        "Year": np.arange(1960, 2021, 5),
        "Population": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "GDP": [30, 35, 40, 46, 52, 58, 65, 72, 80, 88, 96, 105, 115]
    }),
    "Ecuador": pd.DataFrame({
        "Year": np.arange(1960, 2021, 5),
        "Population": [5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11],
        "GDP": [20, 22, 24, 26, 28, 30, 32, 35, 38, 41, 44, 47, 50]
    }),
    "Guatemala": pd.DataFrame({
        "Year": np.arange(1960, 2021, 5),
        "Population": [3, 3.2, 3.5, 3.8, 4, 4.3, 4.6, 4.9, 5.2, 5.5, 5.8, 6.1, 6.5],
        "GDP": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24]
    }),
    "Dominican Republic": pd.DataFrame({
        "Year": np.arange(1960, 2021, 5),
        "Population": [2, 2.2, 2.5, 2.8, 3, 3.3, 3.6, 3.9, 4.2, 4.5, 4.8, 5.1, 5.5],
        "GDP": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    }),
}

categories = ["Population", "GDP"]

# --------------------------
# USER INPUTS
# --------------------------
category = st.selectbox("Select a data category:", categories)
degree = st.slider("Select polynomial regression degree:", 3, 8, 3)
increment = st.slider("Graph increments (years):", 1, 10, 1)
extrapolate_years = st.slider("Extrapolate into future (years):", 0, 20, 5)
countries = st.multiselect("Select countries to analyze:", list(data_samples.keys()), default=list(data_samples.keys()))

# --------------------------
# FUNCTION ANALYSIS
# --------------------------
def function_analysis(model, poly, years, country, category):
    X = poly.transform(years.reshape(-1,1))
    y_pred = model.predict(X)
    dy = np.gradient(y_pred, years)
    ddy = np.gradient(dy, years)
    
    # Max/Min
    max_idx = np.argmax(y_pred)
    min_idx = np.argmin(y_pred)
    
    sentences = []
    sentences.append(f"The {category} of {country} reached a local maximum of {round(y_pred[max_idx],2)} in {years[max_idx]}.")
    sentences.append(f"The {category} of {country} reached a local minimum of {round(y_pred[min_idx],2)} in {years[min_idx]}.")
    
    # Increasing/Decreasing trends
    inc = years[dy>0]
    dec = years[dy<0]
    if len(inc) > 0:
        sentences.append(f"The {category} was generally increasing between {inc[0]} and {inc[-1]}.")
    if len(dec) > 0:
        sentences.append(f"The {category} was generally decreasing between {dec[0]} and {dec[-1]}.")
    
    # Fastest growth/decline
    max_growth_idx = np.argmax(dy)
    max_decline_idx = np.argmin(dy)
    sentences.append(f"The {category} was growing fastest at {round(dy[max_growth_idx],2)} units/year in {years[max_growth_idx]}.")
    sentences.append(f"The {category} was declining fastest at {round(dy[max_decline_idx],2)} units/year in {years[max_decline_idx]}.")
    
    # Domain/Range
    sentences.append(f"The domain (years) for {country} is {years[0]} to {years[-1]}.")
    sentences.append(f"The range (predicted {category}) for {country} is {round(min(y_pred),2)} to {round(max(y_pred),2)}.")
    
    return sentences

# --------------------------
# TABS
# --------------------------
tabs = st.tabs(["Raw Data","Regression & Plot","Function Analysis","Prediction","Avg Rate of Change","US Latin Comparison","Report"])

# --- Raw Data ---
with tabs[0]:
    st.subheader("Raw Data Table (Editable)")
    for c in countries:
        st.write(f"### {c}")
        st.data_editor(data_samples[c][["Year", category]], num_rows="dynamic")

# --- Regression & Plot ---
with tabs[1]:
    st.subheader("Regression Plot & Equations")
    fig, ax = plt.subplots(figsize=(10,6))
    for c in countries:
        df = data_samples[c]
        X = df["Year"].values.reshape(-1,1)
        y = df[category].values
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        
        # Regression & extrapolation
        years = np.arange(df["Year"].min(), df["Year"].max()+extrapolate_years+1, increment)
        y_pred = model.predict(poly.transform(years.reshape(-1,1)))
        ax.scatter(X, y, label=f"{c} data")
        ax.plot(years, y_pred, label=f"{c} regression")
        
        # Equation
        coefs = model.coef_
        intercept = model.intercept_
        terms = [f"{round(coefs[i],2)}*x^{i}" for i in range(len(coefs))]
        equation = " + ".join(terms) + f" + {round(intercept,2)}"
        st.markdown(f"**{c} Regression Equation:** {equation}")
    ax.set_xlabel("Year")
    ax.set_ylabel(category)
    ax.legend()
    st.pyplot(fig)

# --- Function Analysis ---
with tabs[2]:
    st.subheader("Function Analysis & Conjectures")
    for c in countries:
        df = data_samples[c]
        X = df["Year"].values.reshape(-1,1)
        y = df[category].values
        poly = PolynomialFeatures(degree=degree)
        model = LinearRegression().fit(poly.fit_transform(X), y)
        years_full = np.arange(df["Year"].min(), df["Year"].max()+1)
        sentences = function_analysis(model, poly, years_full, c, category)
        for s in sentences:
            st.write(s)

# --- Prediction ---
with tabs[3]:
    st.subheader("Prediction / Interpolation / Extrapolation")
    pred_year = st.number_input("Enter year to predict:", min_value=1950, max_value=2100, value=2030)
    for c in countries:
        df = data_samples[c]
        X = df["Year"].values.reshape(-1,1)
        y = df[category].values
        poly = PolynomialFeatures(degree=degree)
        model = LinearRegression().fit(poly.fit_transform(X), y)
        pred_val = model.predict(poly.transform([[pred_year]]))[0]
        st.write(f"Predicted {category} for {c} in {pred_year}: {round(pred_val,2)}")

# --- Average Rate of Change ---
with tabs[4]:
    st.subheader("Average Rate of Change")
    y1 = st.number_input("Start year:", min_value=1950, max_value=2100, value=1960, key="start")
    y2 = st.number_input("End year:", min_value=1950, max_value=2100, value=2020, key="end")
    if y2>y1:
        for c in countries:
            df = data_samples[c]
            X = df["Year"].values.reshape(-1,1)
            y = df[category].values
            poly = PolynomialFeatures(degree=degree)
            model = LinearRegression().fit(poly.fit_transform(X), y)
            val1 = model.predict(poly.transform([[y1]]))[0]
            val2 = model.predict(poly.transform([[y2]]))[0]
            avg_rate = (val2-val1)/(y2-y1)
            st.write(f"Average rate of change for {c} between {y1}-{y2}: {round(avg_rate,2)} units/year")

# --- US Latin Comparison (Illustrative) ---
with tabs[5]:
    st.subheader("US Latin Groups (Illustrative)")
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

# --- Report ---
with tabs[6]:
    st.subheader("Printer-Friendly Report")
    st.download_button("🖨️ Download Report", "Report with regression analysis, function analysis, predictions, and charts. By Racely Ortega", file_name="report.txt")
