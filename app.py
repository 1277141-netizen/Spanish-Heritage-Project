import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import wbdata
import datetime

# --------------------------
# PAGE SETUP
# --------------------------
st.set_page_config(page_title="Latin America Regression Explorer", layout="wide")
st.title("📊 Latin America Regression Explorer")
st.write("Analyze historical Latin American data with polynomial regression and function analysis.\n**By Racely Ortega**")

# --------------------------
# CONFIG
# --------------------------
latin_countries = {
    "Brazil": "BRA",
    "Mexico": "MEX",
    "Argentina": "ARG"
}

indicators = {
    "Population": "SP.POP.TOTL",
    "Unemployment rate": "SL.UEM.TOTL.ZS",
    "Education levels (0-25)": "SE.SEC.CUAT.UP.ZS",  # proxy
    "Life expectancy": "SP.DYN.LE00.IN",
    "Average wealth": "NY.GNP.PCAP.CD",
    "Average income": "NY.GDP.PCAP.CD",
    "Birth rate": "SP.DYN.CBRT.IN",
    "Immigration out of the country": "SM.POP.NETM",
    "Murder Rate": "VC.IHR.PSRC.P5"
}

# Synthetic US Latin group data (illustrative)
us_groups = {
    "Mexican-Americans": np.random.randint(50, 90, 70),
    "Puerto Ricans": np.random.randint(55, 85, 70),
    "Cuban-Americans": np.random.randint(60, 95, 70)
}

# --------------------------
# THREAD-SAFE DATA FETCHING
# --------------------------
def fetch_wb_data(country_code, indicator_code):
    try:
        df = wbdata.get_dataframe({indicator_code: 'Value'}, country=country_code)
        df = df.reset_index()
        df.rename(columns={'date':'Date', indicator_code:'Value'}, inplace=True)
        df['Year'] = pd.to_datetime(df['Date']).dt.year
        current_year = datetime.datetime.now().year
        df = df[df['Year'] >= current_year - 70]
        df = df.sort_values("Year")
        df = df.dropna()
        if len(df) < 30:
            return None
        return df[['Year','Value']]
    except Exception as e:
        st.warning(f"Error fetching data for {country_code}: {e}")
        return None

# --------------------------
# USER INPUTS
# --------------------------
category = st.selectbox("Select data category:", list(indicators.keys()))
degree = st.slider("Select polynomial regression degree:", 3, 8, 3)
increment = st.slider("Graph increments (years):", 1, 10, 1)
extrapolate_years = st.slider("Extrapolate into future (years):", 0, 50, 10)

# Fetch data for all countries, filter those with enough data
frames = {}
for c, code in latin_countries.items():
    df = fetch_wb_data(code, indicators[category])
    if df is not None:
        df.rename(columns={'Value': category}, inplace=True)
        frames[c] = df

if not frames:
    st.error("No countries have sufficient data for this category. Please try another.")
    st.stop()

country_options = list(frames.keys())
countries = st.multiselect("Select countries to analyze:", country_options, default=country_options)

# --------------------------
# TABS FOR CLEAR FUNCTION SECTIONS
# --------------------------
tabs = st.tabs([
    "Raw Data", 
    "Regression Plot & Equation", 
    "Function Analysis", 
    "Prediction / Extrapolation", 
    "Average Rate of Change", 
    "US Latin Comparison", 
    "Printer-Friendly Report"
])

# --------------------------
# RAW DATA TAB
# --------------------------
with tabs[0]:
    st.subheader("Raw Data Table (Editable)")
    first_country = countries[0]
    edited_df = st.data_editor(frames[first_country], num_rows="dynamic")
    st.write("Inputs: Year | Outputs:", category)

# --------------------------
# REGRESSION PLOT TAB
# --------------------------
with tabs[1]:
    st.subheader("Regression Plot & Equations")
    fig, ax = plt.subplots(figsize=(10,6))
    for c in countries:
        df = frames[c]
        X = df["Year"].values.reshape(-1,1)
        y = df[category].values
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)

        # Prediction + extrapolation
        years = np.arange(df["Year"].min(), df["Year"].max() + extrapolate_years + 1, increment).reshape(-1,1)
        preds = model.predict(poly.transform(years))

        # Scatter & regression curve
        ax.scatter(X, y, label=f"{c} data")
        ax.plot(years, preds, label=f"{c} regression", linestyle='--' if extrapolate_years>0 else '-')

        # Regression equation
        coefs = model.coef_
        intercept = model.intercept_
        terms = [f"{round(coefs[i],2)}*x^{i}" for i in range(len(coefs))]
        equation = " + ".join(terms) + f" + {round(intercept,2)}"
        st.markdown(f"**{c} Regression Equation (degree {degree}):** {equation}")

    ax.set_xlabel("Year")
    ax.set_ylabel(category)
    ax.legend()
    st.pyplot(fig)

# --------------------------
# FUNCTION ANALYSIS TAB
# --------------------------
with tabs[2]:
    st.subheader("Function Analysis & Conjectures")
    st.info("Analyze maxima, minima, increasing/decreasing trends, domain, and range with real-world meaning. Example: 'The population of Brazil reached a local maximum on 11/2/2017. The population was growing fastest in the 1960s due to economic expansion.'")

# --------------------------
# PREDICTION / EXTRAPOLATION TAB
# --------------------------
with tabs[3]:
    st.subheader("Prediction / Interpolation / Extrapolation")
    pred_year = st.number_input("Enter year to predict:", min_value=1950, max_value=2100, value=2035)
    for c in countries:
        df = frames[c]
        X = df["Year"].values.reshape(-1,1)
        y = df[category].values
        poly = PolynomialFeatures(degree=degree)
        model = LinearRegression().fit(poly.fit_transform(X), y)
        pred = model.predict(poly.transform([[pred_year]]))[0]
        st.write(f"In {pred_year}, predicted {category} for {c}: {round(pred,2)}")

# --------------------------
# AVERAGE RATE OF CHANGE TAB
# --------------------------
with tabs[4]:
    st.subheader("Average Rate of Change")
    y1 = st.number_input("Start year:", min_value=1950, max_value=2100, value=1960, key="y1")
    y2 = st.number_input("End year:", min_value=1950, max_value=2100, value=2020, key="y2")
    if y2 > y1:
        for c in countries:
            df = frames[c]
            X = df["Year"].values.reshape(-1,1)
            y = df[category].values
            poly = PolynomialFeatures(degree=degree)
            model = LinearRegression().fit(poly.fit_transform(X), y)
            val1 = model.predict(poly.transform([[y1]]))[0]
            val2 = model.predict(poly.transform([[y2]]))[0]
            avg_rate = (val2 - val1)/(y2-y1)
            st.write(f"Avg rate of change for {c} between {y1}-{y2}: {round(avg_rate,2)} units/year")

# --------------------------
# US LATIN COMPARISON TAB
# --------------------------
with tabs[5]:
    st.subheader("US Latin Groups Comparison (Illustrative)")
    show_us = st.checkbox("Compare with Latin groups in the US")
    if show_us:
        fig2, ax2 = plt.subplots()
        years = np.arange(1955,2025)
        for g, vals in us_groups.items():
            ax2.plot(years, vals, label=g)
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Index Value")
        ax2.legend()
        st.pyplot(fig2)

# --------------------------
# PRINTER-FRIENDLY REPORT TAB
# --------------------------
with tabs[6]:
    st.subheader("Printer-Friendly Report")
    st.download_button("Download Printer-Friendly Report", "Analysis report goes here", file_name="report.txt")
