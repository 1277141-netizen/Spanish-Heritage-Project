import streamlit as st
import pandas as pd
import numpy as np
import datetime

try:
    import matplotlib.pyplot as plt
except Exception as e:
    st.warning("Matplotlib could not be loaded. Graphs will not display.")
    plt = None

try:
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
except Exception as e:
    st.warning("scikit-learn not available. Regression features disabled.")
    PolynomialFeatures = None
    LinearRegression = None

import wbdata

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

if plt and PolynomialFeatures and LinearRegression:
    fig, ax = plt.subplots(figsize=(10, 6))
else:
    ax = None

if PolynomialFeatures and LinearRegression:
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
st.write("Make your own conjectures about why the entity had significant changes during a given time period.")

st.subheader("🔮 Prediction & Extrapolation")
if PolynomialFeatures and LinearRegression:
    pred_year = st.number_input("Enter a year to predict:", min_value=1950, max_value=2100, value=2035)
    for c, df in frames.items():
        X = df["year"].values.reshape(-1, 1)
        y = df[category].values
        poly = PolynomialFeatures(degree=degree)
        model = LinearRegression().fit(poly.fit_transform(X), y)
        pred = model.predict(poly.transform([[pred_year]]))[0]
        st.write(f"According to the regression model, in {pred_year}, {c}'s {category.lower()} ≈ {round(pred, 2)}.")

st.markdown("---")
st.markdown("### 📚 Project Instructions & Grading Information")
st.write("- Make conjectures about why your country/entity experienced changes.")
st.write("- Include a prediction beyond the dataset (extrapolation).")
st.write("- Relate to national identity and economy for higher scores.")
st.write("- Counts as 35% of your grade under performance tasks.")
st.write("- Reserve your country/entity in the provided Google Sheet.")
st.markdown("<h5 style='text-align: center;'>By Racely Ortega</h5>", unsafe_allow_html=True)
st.caption("Data source: World Bank (via wbdata). Application built for educational use.")
