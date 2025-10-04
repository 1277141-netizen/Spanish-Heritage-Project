import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import datetime

st.set_page_config(layout="wide", page_title="Latin Countries Regression Explorer")

AUTHOR = "By Racely Ortega"

# -----------------------------
# Categories and CSV file map
# -----------------------------
DATA_FILES = {
    "Population": "data/population.csv",
    "Unemployment rate": "data/unemployment.csv",
    "Education (0–25 scale)": "data/education.csv",
    "Life expectancy": "data/life_expectancy.csv",
    "Average wealth": "data/wealth.csv",
    "Average income": "data/income.csv",
    "Birth rate": "data/birth_rate.csv",
    "Immigration": "data/immigration.csv",
    "Murder rate": "data/murder_rate.csv",
}

COUNTRIES = ["Argentina","Brazil","Chile","Colombia","Costa Rica",
             "Mexico","Panama","Peru","Uruguay","Guyana"]

# -----------------------------
# Helper functions
# -----------------------------
def load_data(category):
    path = DATA_FILES[category]
    df = pd.read_csv(path)
    return df

def poly_fit(years, values, degree):
    mask = ~np.isnan(values)
    x = np.array(years)[mask].reshape(-1,1)
    y = np.array(values)[mask]
    if len(x) < degree+1:
        return None, None
    model = make_pipeline(PolynomialFeatures(degree, include_bias=True), LinearRegression())
    model.fit(x, y)
    coefs = np.polyfit(x.flatten(), y, degree)
    p = np.poly1d(coefs)
    return p, coefs

def format_equation(coefs):
    terms = []
    degree = len(coefs)-1
    for i, c in enumerate(coefs):
        power = degree - i
        if abs(c) < 1e-12:
            continue
        coeff = f"{c:.6g}"
        if power == 0:
            terms.append(f"{coeff}")
        elif power == 1:
            terms.append(f"{coeff}*x")
        else:
            terms.append(f"{coeff}*x**{power}")
    return "f(x) = " + " + ".join(terms)

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Controls")
category = st.sidebar.selectbox("Select Category", list(DATA_FILES.keys()))
degree = st.sidebar.slider("Polynomial Degree", 3, 8, 3)
start_year = st.sidebar.number_input("Start Year", 1955, 2100, 1955)
end_year = st.sidebar.number_input("End Year", 1955, 2100, 2024)
extrapolate_years = st.sidebar.number_input("Extrapolate Years", 0, 50, 10)
year_increment = st.sidebar.slider("Graph Year Increment", 1, 10, 5)
compare_countries = st.sidebar.multiselect("Select Countries", COUNTRIES, default=COUNTRIES[:3])

interpolation_year = st.sidebar.number_input("Interpolate/Extrapolate Value Year", start_year, end_year+extrapolate_years, start_year)
roc_start_year = st.sidebar.number_input("Rate of Change Start Year", start_year, end_year, start_year)
roc_end_year = st.sidebar.number_input("Rate of Change End Year", start_year, end_year, end_year)

# -----------------------------
# Load Data
# -----------------------------
df = load_data(category)
df = df[df['country'].isin(compare_countries)]
df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]

st.subheader("Raw Data (Editable)")
edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
df = edited_df.copy()

# -----------------------------
# Regression & Plotting
# -----------------------------
st.subheader("Regression & Analysis")
fig, ax = plt.subplots(figsize=(12,6))
results = []

for country in compare_countries:
    df_c = df[df['country']==country].sort_values('year')
    years = df_c['year'].values
    values = df_c['value'].values

    p, coefs = poly_fit(years, values, degree)
    if p is None:
        st.warning(f"Not enough data points for {country}.")
        continue

    # Scatter plot
    ax.scatter(years, values, label=f"{country} data")

    # Fit line
    x_fit = np.arange(start_year, end_year+1)
    y_fit = p(x_fit)
    ax.plot(x_fit, y_fit, label=f"{country} fit")

    # Extrapolation
    if extrapolate_years > 0:
        x_ext = np.arange(end_year+1, end_year+extrapolate_years+1)
        y_ext = p(x_ext)
        ax.plot(x_ext, y_ext, linestyle="--", label=f"{country} projection")

    results.append({
        'country': country,
        'equation': format_equation(coefs),
        'poly': p
    })

ax.set_xlabel("Year")
ax.set_ylabel(category)
ax.set_title(f"{category} Regression (Degree {degree})")
ax.legend()
plt.xticks(np.arange(start_year, end_year+extrapolate_years+1, year_increment), rotation=45)
st.pyplot(fig)

# -----------------------------
# Function Analysis
# -----------------------------
st.subheader("Function Analysis")
for res in results:
    st.markdown(f"### {res['country']}")
    st.markdown(f"**Equation:** `{res['equation']}`")

    # Interpolation/Extrapolation
    val = res['poly'](interpolation_year)
    st.write(f"Value at year {interpolation_year}: {val:.2f}")

    # Average rate of change
    y1 = res['poly'](roc_start_year)
    y2 = res['poly'](roc_end_year)
    roc = (y2 - y1) / (roc_end_year - roc_start_year)
    st.write(f"Average rate of change from {roc_start_year} to {roc_end_year}: {roc:.2f} per year")

st.markdown(f"---\n{AUTHOR}")
