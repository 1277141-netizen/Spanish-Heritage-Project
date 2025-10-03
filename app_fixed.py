import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

st.title("Latin America Data Regression App")
st.subheader("Created by Racely Ortega")

st.markdown("This app analyzes historical data from Latin American countries with polynomial regression models.")

# Example dummy dataset (replace with real CSVs later)
years = np.arange(1950, 2021)
values = np.random.rand(len(years)) * 100

df = pd.DataFrame({
    "Year": years,
    "Population": values
})

st.write("### Raw Data (Editable)")
edited_df = st.data_editor(df, num_rows="dynamic")

degree = st.slider("Select regression degree", min_value=3, max_value=10, value=3)
interval = st.slider("Select graph interval (years)", min_value=1, max_value=10, value=1)

X = edited_df["Year"].values.reshape(-1, 1)
y = edited_df["Population"].values

model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model.fit(X, y)

# Predict across the range
X_pred = np.arange(edited_df["Year"].min(), edited_df["Year"].max() + 1, interval).reshape(-1, 1)
y_pred = model.predict(X_pred)

# Display equation coefficients
coeffs = model.named_steps['linearregression'].coef_
intercept = model.named_steps['linearregression'].intercept_
st.write("### Regression Model Equation")
st.write(f"Intercept: {intercept:.4f}")
st.write(f"Coefficients: {coeffs}")

# Plot
fig, ax = plt.subplots()
ax.scatter(X, y, color="blue", label="Data")
ax.plot(X_pred, y_pred, color="red", label="Regression Curve")
ax.set_xlabel("Year")
ax.set_ylabel("Population")
ax.legend()
st.pyplot(fig)

st.markdown("---")
st.caption("App created by Racely Ortega")
