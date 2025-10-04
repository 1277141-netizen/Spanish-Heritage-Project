import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wbdata
import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Latin America Regression Explorer", layout="wide")

st.title("📊 Latin America Regression Explorer")
st.write("Analyze historical Latin American data with polynomial regression and function analysis.\n**By Racely Ortega**")

# --------------------------------
# CONFIG
# --------------------------------
latin_countries = {
    "Brazil": "BRA",
    "Mexico": "MEX",
    "Argentina": "ARG",
    "Colombia": "COL",
    "Chile": "CHL",
    "Peru": "PER",
    "Venezuela": "VEN",
    "Ecuador": "ECU",
    "Guatemala": "GTM",
    "Dominican Republic": "DOM",
}

indicators = {
    "Population": "SP.POP.TOTL",
    "Unemployment rate": "SL.UEM.TOTL.ZS",
    "Education levels (0-25 proxy: school years)": "SE.SEC.CUAT.UP.ZS",
    "Life expectancy": "SP.DYN.LE00.IN",
    "Average wealth (GNI per capita)": "NY.GNP.PCAP.CD",
    "Average income (GDP per capita)": "NY.GDP.PCAP.CD",
    "Birth rate": "SP.DYN.CBRT.IN",
    "Immigration out of the country (net migration)": "SM.POP.NETM",
    "Murder Rate": "VC.IHR.PSRC.P5",
}

# Synthetic US Latin group data (illustrative placeholder)
us_groups = {
    "Mexican-Americans": np.random.randint(50, 90, 70),
    "Puerto Ricans": np.random.randint(55, 85, 70),
    "Cuban-Americans": np.random.randint(60, 95, 70),
}

# --------------------------------
# FETCH HISTORICAL DATA
# --------------------------------
@st.cache_data
def get_wb_data_fixed(country_code, indicator_code):
    try:
        df = wbdata.get_dataframe({indicator_code: 'Value'}, country=country_code)
        df = df.reset_index()
        df.rename(columns={'date':'Date', indicator_code:'Value'}, inplace=True)
        df['Year'] = pd.to_datetime(df['Date']).dt.year
        current_year = datetime.datetime.now().year
        df = df[df['Year'] >= current_year - 70]
        df = df.sort_values("Year")
        df = df.dropna()
        return df[['Year', 'Value']]
    except Exception as e:
        st.error(f"Error fetching data for {country_code}: {e}")
        return None

# --------------------------------
# UI
# --------------------------------
category = st.selectbox("Select a data category:", list(indicators.keys()))
countries = st.multiselect("Select countries to analyze:", list(latin_countries.keys()), default=["Brazil"])

# Fetch data
frames = {}
for c in countries:
    df = get_wb_data_fixed(latin_countries[c], indicators[category])
    if df is not None and not df.empty:
        df.rename(columns={'Value': category}, inplace=True)
        frames[c] = df

# Editable table (first selected country)
if countries and frames:
    st.subheader("Raw Data (Editable)")
    edited_df = st.data_editor(frames[countries[0]], num_rows="dynamic")
    st.write("Inputs: year | Outputs:", category)

# Polynomial regression degree
degree = st.slider("Select polynomial regression degree:", 3, 8, 3)
increment = st.slider("Graph increments (years):", 1, 10, 1)
extrapolate_years = st.slider("Extrapolate into the future (years):", 0, 50, 10)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for c, df in frames.items():
    X = df["Year"].values.reshape(-1,1)
    y = df[category].values
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)

    # Fit line
    years = np.arange(df["Year"].min(), df["Year"].max()+extrapolate_years+1, increment).reshape(-1,1)
    preds = model.predict(poly.transform(years))

    # Scatter & regression
    ax.scatter(df["Year"], y, label=f"{c} data")
    ax.plot(years, preds, label=f"{c} regression")

    # Equation
    coefs = model.coef_
    intercept = model.intercept_
    terms = [f"{round(coefs[i],2)}*x^{i}" for i in range(len(coefs))]
    equation = " + ".join(terms) + f" + {round(intercept,2)}"
    st.markdown(f"**{c} Regression Equation (degree {degree}):** {equation}")

ax.set_xlabel("Year")
ax.set_ylabel(category)
ax.legend()
st.pyplot(fig)

# Analysis section
st.subheader("📈 Function Analysis")
st.write("Interpreting maxima, minima, growth/decline, domain, range with real-world meaning...")
st.info("Example: The population of Brazil reached a local maximum in 2017. The population was growing fastest in the 1960s due to economic expansion...")

# Prediction tool
st.subheader("🔮 Prediction & Interpolation/Extrapolation")
pred_year = st.number_input("Enter a year to predict:", min_value=1950, max_value=2100, value=2035)
for c, df in frames.items():
    X = df["Year"].values.reshape(-1,1)
    y = df[category].values
    poly = PolynomialFeatures(degree=degree)
    model = LinearRegression().fit(poly.fit_transform(X), y)
    pred = model.predict(poly.transform([[pred_year]]))[0]
    st.write(f"In {pred_year}, predicted {category} for {c}: {round(pred,2)}")

# Average rate of change
st.subheader("📐 Average Rate of Change")
y1 = st.number_input("Start year:", min_value=1950, max_value=2100, value=1960)
y2 = st.number_input("End year:", min_value=1950, max_value=2100, value=2020)
if y2 > y1:
    for c, df in frames.items():
        X = df["Year"].values.reshape(-1,1)
        y = df[category].values
        poly = PolynomialFeatures(degree=degree)
        model = LinearRegression().fit(poly.fit_transform(X), y)
        val1 = model.predict(poly.transform([[y1]]))[0]
        val2 = model.predict(poly.transform([[y2]]))[0]
        avg_rate = (val2 - val1)/(y2-y1)
        st.write(f"Avg rate of change for {c} between {y1}-{y2}: {round(avg_rate,2)} units/year")

# US Latin group comparison
st.subheader("🇺🇸 Latin Groups in the U.S. (Illustrative)")
compare_us = st.checkbox("Show comparison with Latin groups in U.S.")
if compare_us:
    fig2, ax2 = plt.subplots()
    years = np.arange(1955, 2025)
    for g, vals in us_groups.items():
        ax2.plot(years, vals, label=g)
    ax2.legend()
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Index Value")
    st.pyplot(fig2)

# Print option
st.download_button("🖨️ Download Printer-Friendly Report", "Analysis report goes here", file_name="report.txt")
