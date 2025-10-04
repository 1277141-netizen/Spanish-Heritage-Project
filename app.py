import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import wbdata
import datetime

st.set_page_config(page_title="Latin Wealth Regression Analysis", layout="wide")

st.title("Regression Analysis of Latin American Economic & Social Indicators")
st.write("Analyze real historical data across the wealthiest Latin countries and perform polynomial regression with full function analysis.\n**By Racely Ortega**")

# Top 10 wealthiest Latin countries (by GDP)
countries = {
    "Brazil": "BRA",
    "Mexico": "MEX",
    "Argentina": "ARG",
    "Chile": "CHL",
    "Colombia": "COL",
    "Peru": "PER",
    "Venezuela": "VEN",
    "Ecuador": "ECU",
    "Dominican Republic": "DOM",
    "Uruguay": "URY"
}

# Category to World Bank indicator mapping
indicators = {
    "Population": "SP.POP.TOTL",
    "Unemployment rate": "SL.UEM.TOTL.ZS",
    "Education levels from 0-25": "SE.SEC.ENRR",
    "Life expectancy": "SP.DYN.LE00.IN",
    "Average wealth": "NY.GNP.PCAP.CD",
    "Average income": "NY.ADJ.NNTY.PC.CD",
    "Birth rate": "SP.DYN.CBRT.IN",
    "Immigration out of the country": "SM.EMI.TOTL.ZS",
    "Murder Rate": "VC.IHR.PSRC.P5"
}

# Function to fetch historical data (70 years)
def fetch_wbdata(country_code, indicator_code):
    try:
        data = wbdata.get_dataframe(
            {indicator_code: 'value'},
            country=country_code,
            convert_date=True
        ).sort_index()
        # Filter last 70 years
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=70*365)
        data = data[start_date:end_date]
        return data
    except Exception as e:
        st.error(f"Error fetching data for {country_code}: {e}")
        return None

# Sidebar controls
selected_indicator = st.sidebar.selectbox("Select Category", list(indicators.keys()))
selected_countries = st.sidebar.multiselect("Select Countries", list(countries.keys()), default=["Mexico"])
degree = st.sidebar.slider("Select Degree of Regression (≥3)", min_value=3, max_value=10, value=3)
increment = st.sidebar.slider("Graph increment in years", 1, 10, 1)

# Data handling
indicator_code = indicators[selected_indicator]
all_data = {}

for country in selected_countries:
    df = fetch_wbdata(countries[country], indicator_code)
    if df is not None and not df.empty:
        df = df.reset_index()
        df.columns = ["Year", "Value"]
        df["Year"] = pd.to_datetime(df["Year"]).dt.year
        df = df.dropna()
        all_data[country] = df

if all_data:
    st.subheader("Raw Data Table (Editable)")
    combined = pd.concat(all_data, names=["Country", "Index"]).reset_index(level=0)
    edited = st.data_editor(combined, num_rows="dynamic")

    # Plot regression for each selected country
    plt.figure(figsize=(10, 6))
    for country, df in all_data.items():
        X = df["Year"].values.reshape(-1, 1)
        y = df["Value"].values
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)

        # Regression line
        years = np.arange(df["Year"].min(), df["Year"].max() + 1, increment).reshape(-1, 1)
        predictions = model.predict(poly.transform(years))

        plt.scatter(X, y, label=f"{country} Data")
        plt.plot(years, predictions, label=f"{country} Fit")

        # Regression equation
        coeffs = model.coef_
        intercept = model.intercept_
        st.markdown(f"### {country} Regression Equation (Degree {degree})")
        eq_terms = " + ".join([f"{coeffs[i]:.4e}x^{i}" for i in range(len(coeffs))])
        st.write(f"**y = {intercept:.4e} + {eq_terms}**")

        # Extrapolation for 20 years
        future_years = np.arange(df["Year"].max() + 1, df["Year"].max() + 21).reshape(-1, 1)
        future_predictions = model.predict(poly.transform(future_years))
        plt.plot(future_years, future_predictions, "--", label=f"{country} Extrapolated")

        st.write(f"The model suggests that the **{selected_indicator.lower()}** in {country} has varied over time.")

    plt.xlabel("Year")
    plt.ylabel(selected_indicator)
    plt.title(f"{selected_indicator} Over Time - Polynomial Regression (Degree {degree})")
    plt.legend()
    st.pyplot(plt)

    st.subheader("Interpretation and Conjectures")
    st.write("""
Make your own conjectures as to why the entity had significant changes during a given time period.

A prediction of a function output value must be given for an input value that is beyond the data set (extrapolation). Use proper units when describing rates.

Example: "According to the regression model, the rate of crime in the county will approach a rate of 100 serious felonies per year in 2050."

More points will be given for problems that relate strongly to the national identity and/or the economy of the country.

This project will be counted as a performance task and will go into the grade book with a category weight of 35% of your grade.

Students should pick their own entity to focus their conjectures on (no duplicates).
""")
else:
    st.error("No valid data available from World Bank for the selected category/countries.")
