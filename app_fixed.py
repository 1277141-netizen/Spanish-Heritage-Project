import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import datetime

# Matplotlib import wrapped in a safe block
try:
    import matplotlib.pyplot as plt
except Exception as e:
    st.warning("Matplotlib could not be loaded. Graphs will not display.")
    plt = None

import wbdata

# App config
st.set_page_config(page_title="Latin America Regression Explorer", layout="wide")

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
    "Education levels (0-25 proxy: secondary completion %)": "SE.SEC.CUAT.UP.ZS",
    "Life expectancy": "SP.DYN.LE00.IN",
    "Average wealth (GNI per capita)": "NY.GNP.PCAP.CD",
    "Average income (GDP per capita)": "NY.GDP.PCAP.CD",
    "Birth rate": "SP.DYN.CBRT.IN",
    "Immigration out of the country (net migration)": "SM.POP.NETM",
    "Murder Rate": "VC.IHR.PSRC.P5",
}

us_groups = {
    "Mexican-Americans": np.random.randint(50, 90, 70),
    "Puerto Ricans": np.random.randint(55, 85, 70),
    "Cuban-Americans": np.random.randint(60, 95, 70),
}

@st.cache_data
def get_wb_data(country_code, indicator_code):
    try:
        start_date = datetime.datetime(1955, 1, 1)
        end_date = datetime.datetime(2025, 1, 1)
        df = wbdata.get_dataframe(
            {indicator_code: indicator_code},
            country=country_code,
            data_date=(start_date, end_date),
            convert_date=True,
        )
        df.reset_index(inplace=True)
        df["year"] = df["date"].dt.year
        df = df.sort_values("year")
        df = df.dropna()
        if df.empty:
            return None
        return df[["year", indicator_code]]
    except Exception as e:
        st.warning(f"⚠️ Could not fetch data for {country_code}: {e}")
        return None

st.title("📊 Latin America Regression Explorer")

category = st.selectbox("Select a data category:", list(indicators.keys()))
countries = st.multiselect("Select countries to analyze:", list(latin_countries.keys()), default=["Brazil"])

frames = {}
for c in countries:
    df = get_wb_data(latin_countries[c], indicators[category])
    if df is not None:
        df.rename(columns={indicators[category]: category}, inplace=True)
        frames[c] = df

if not frames:
    st.error("No valid data available from World Bank for the selected category/countries.")
    st.stop()

st.subheader("Raw Data (Editable)")
first_country = list(frames.keys())[0]
edited_df = st.data_editor(frames[first_country], num_rows="dynamic")
st.caption("Inputs: year | Outputs: " + category)

degree = st.slider("Polynomial Regression Degree:", 3, 8, 3)
increment = st.slider("Graph increments (years):", 1, 10, 1)
extrapolate_years = st.slider("Extrapolate into the future (years):", 0, 50, 10)

if plt:
    fig, ax = plt.subplots(figsize=(10, 6))
else:
    ax = None

for c, df in frames.items():
    X = df["year"].values.reshape(-1, 1)
    y = df[category].values

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)

    years = np.arange(df["year"].min(), df["year"].max() + extrapolate_years, increment).reshape(-1, 1)
    preds = model.predict(poly.transform(years))

    if plt:
        ax.scatter(df["year"], y, label=f"{c} data")
        ax.plot(years, preds, label=f"{c} regression")

    # Display equation
    coefs = model.coef_
    intercept = model.intercept_
    terms = [f"{round(coefs[i], 3)}*x^{i}" for i in range(len(coefs))]
    equation = " + ".join(terms) + f" + {round(intercept, 3)}"
    st.markdown(f"**{c} Regression Equation (degree {degree}):** `{equation}`")

if plt:
    ax.set_xlabel("Year")
    ax.set_ylabel(category)
    ax.legend()
    st.pyplot(fig)

st.subheader("📈 Function Analysis & Conjectures")
st.write("""
Make your own **conjectures** as to why the entity had significant changes during a given time period.  
You may consider social, political, or economic events that could explain increases, decreases, or irregular trends.  
This helps connect mathematical models with **real-world context** and national identity.
""")

st.subheader("🔮 Prediction & Extrapolation")
pred_year = st.number_input("Enter a year to predict:", min_value=1950, max_value=2100, value=2035)
for c, df in frames.items():
    X = df["year"].values.reshape(-1, 1)
    y = df[category].values
    poly = PolynomialFeatures(degree=degree)
    model = LinearRegression().fit(poly.fit_transform(X), y)
    pred = model.predict(poly.transform([[pred_year]]))[0]
    st.write(f"According to the regression model, in **{pred_year}**, {c}'s {category.lower()} will be approximately **{round(pred, 2)}**.")

st.subheader("📐 Average Rate of Change")
y1 = st.number_input("Start year:", min_value=1950, max_value=2100, value=1960)
y2 = st.number_input("End year:", min_value=1950, max_value=2100, value=2020)
if y2 > y1:
    for c, df in frames.items():
        X = df["year"].values.reshape(-1, 1)
        y = df[category].values
        poly = PolynomialFeatures(degree=degree)
        model = LinearRegression().fit(poly.fit_transform(X), y)
        val1 = model.predict(poly.transform([[y1]]))[0]
        val2 = model.predict(poly.transform([[y2]]))[0]
        avg_rate = (val2 - val1) / (y2 - y1)
        st.write(f"The average rate of change for {c} between {y1}-{y2} is approximately **{round(avg_rate, 2)} units per year**.")

st.markdown("---")
st.markdown("""
### 📚 Project Instructions & Grading Information
- Make your own **conjectures** about why your country/entity experienced significant changes.  
- Include a **prediction** beyond the dataset (extrapolation) with proper units.  
- Stronger connections to **national identity and economy** earn more points.  
- This is a **performance task** counting as **35% of your grade** in the performance task category.  
- Each student must pick their own entity — reserve your country/entity in the provided Google Sheet (first come, first serve).  
- Example: *“According to the regression model, the rate of crime in the county will approach 100 serious felonies per year by 2050.”*
""")
st.markdown("<h5 style='text-align: center;'>By Racely Ortega</h5>", unsafe_allow_html=True)
st.caption("Data source: World Bank (via wbdata). Application built for educational use.")
