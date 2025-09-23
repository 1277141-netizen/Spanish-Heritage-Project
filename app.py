"""
Streamlit app: Historical data regression & function analysis for top Latin American countries
Fetches data from World Bank and OurWorldInData (when needed). 
Features:
- Choose a category (population, unemployment, education (mean years of schooling), life expectancy, GDP per capita, birth rate, net migration, homicide rate)
- Choose one or multiple countries (top Latin countries by GDP per capita)
- Editable raw data table (Streamlit's data_editor)
- Fit a polynomial regression of degree >= 3, show equation, plot scatter + fitted curve
- Options: fit increments (plot the regression evaluated every 1-10 years), extrapolate years, interpolation/extrapolation of a value, average rate of change between two years
- Function analysis: local extrema, intervals of increase/decrease, fastest increase/decrease (max derivative magnitude), domain/range
- Export a printer-friendly HTML with results
Notes:
- The app dynamically fetches data from World Bank API and Our World in Data for mean years of schooling.
- Some indicators have limited historical coverage; missing years will be left as NaN and shown in the editable table.
- Requirements: see requirements file in repository root.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import base64
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import optimize
from scipy import stats
from numpy.polynomial import Polynomial

st.set_page_config(layout="wide", page_title="Latin Countries Historical Regression Explorer")

# --- Helper functions ---
WORLD_BANK_BASE = "https://api.worldbank.org/v2"
CURRENT_YEAR = datetime.now().year
DEFAULT_START_YEAR = max(1950, CURRENT_YEAR - 70)  # last 70 years

# Top 10 (by recent GDP per capita ranks among Latin American countries) - ISO2 codes and names
TOP_LATIN = {
    "Panama":"PA",
    "Uruguay":"UY",
    "Chile":"CL",
    "Costa Rica":"CR",
    "Argentina":"AR",
    "Mexico":"MX",
    "Dominican Republic":"DO",
    "Brazil":"BR",
    "Colombia":"CO",
    "Peru":"PE"
}

# Mapping user categories to data sources & indicator codes
CATEGORIES = {
    "Population":"SP.POP.TOTL",
    "Unemployment rate":"SL.UEM.TOTL.ZS",
    "Education (mean years of schooling, 0-25 scale)": "MEAN.YEARS.SCHOOLING",  # special (OWID/Barro-Lee)
    "Life expectancy":"SP.DYN.LE00.IN",
    "Average wealth (GDP per capita, current US$)":"NY.GDP.PCAP.CD",
    "Average income (GDP per capita, current US$)":"NY.GDP.PCAP.CD",
    "Birth rate":"SP.DYN.CBRT.IN",
    "Immigration (net migration)":"SM.POP.NETM",
    "Murder Rate (intentional homicides per 100k)":"VC.IHR.PSRC.P5",
    "Intentional homicides per 100k":"VC.IHR.PSRC.P5"
}

def wb_fetch(country_code, indicator, start=DEFAULT_START_YEAR, end=CURRENT_YEAR):
    \"\"\"Fetch time series for a single country and indicator from World Bank API.
       Returns a pandas Series indexed by year (int).\"\"\"
    if indicator == \"MEAN.YEARS.SCHOOLING\":
        # We'll try Our World in Data Barro-Lee dataset (mean years) via OWID long-run CSV if available
        # OWID endpoint for mean years of schooling long-run (adapted): use ourworldindata grapher raw
        owid_url = \"https://archive.ourworldindata.org/grapher/mean-years-of-schooling-long-run.csv\"
        try:
            r = requests.get(owid_url, timeout=15)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            # OWID columns: Entity, Year, Mean years of schooling
            df = df[df['Entity'].str.lower().isin([country_name.lower() for country_name in TOP_LATIN.keys()]) | (df['Entity'].str.lower() == country_code.lower())]
            # find matching entity
            ent = None
            # match by iso2: OWID has 'ISO code' column sometimes; try to use country name
            if 'ISO' in df.columns or 'iso_code' in df.columns:
                for col in df.columns:
                    if col.lower().startswith('iso') :
                        if any(df[col].astype(str).str.upper()==country_code.upper()):
                            sub = df[df[col].astype(str).str.upper()==country_code.upper()]
                            s = sub.set_index('Year')['Mean years of schooling'] if 'Mean years of schooling' in sub.columns else sub.set_index('Year').iloc[:,0]
                            s.index = s.index.astype(int)
                            return s.loc[start:end]
            # fallback match by country name mapping
            # find mapping from code -> name in TOP_LATIN
            name = None
            for n,c in TOP_LATIN.items():
                if c.upper() == country_code.upper():
                    name = n
                    break
            if name:
                sub = df[df['Entity'].str.lower() == name.lower()]
                if not sub.empty:
                    s = sub.set_index('Year')['Mean years of schooling']
                    s.index = s.index.astype(int)
                    return s.loc[start:end]
        except Exception as e:
            st.warning(f\"Could not fetch education series from OWID: {e}\")
            return pd.Series(dtype='float64')
    # For world bank indicators
    url = f\"{WORLD_BANK_BASE}/country/{country_code}/indicator/{indicator}?date={start}:{end}&per_page=2000&format=json\"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data or len(data) < 2:
            return pd.Series(dtype='float64')
        records = data[1]
        years = {}
        for rec in records:
            yr = rec.get('date')
            val = rec.get('value')
            if yr and val is not None:
                years[int(yr)] = float(val)
        s = pd.Series(years)
        s = s.sort_index()
        return s
    except Exception as e:
        st.error(f\"Error fetching World Bank data: {e}\")
        return pd.Series(dtype='float64')

def fetch_multi(countries, indicator, start=DEFAULT_START_YEAR, end=CURRENT_YEAR):
    df = pd.DataFrame(index=range(start, end+1))
    for country in countries:
        iso = TOP_LATIN.get(country)
        if not iso:
            continue
        s = wb_fetch(iso, indicator, start, end)
        df[country] = s
    return df

def poly_fit_years(x, y, degree=3):
    # remove NaNs
    mask = ~np.isnan(y) & ~np.isnan(x)
    if mask.sum() < degree+1:
        raise ValueError(\"Not enough data points to fit the requested polynomial degree.\")
    coeffs = np.polyfit(x[mask], y[mask], degree)
    p = np.poly1d(coeffs)
    return p, coeffs

def poly_to_equation(coeffs):
    terms = []
    deg = len(coeffs)-1
    for i,c in enumerate(coeffs):
        power = deg - i
        if abs(c) < 1e-12:
            continue
        coef = round(c,6)
        if power == 0:
            terms.append(f\"{coef}\")
        elif power == 1:
            terms.append(f\"{coef}*x\")
        else:
            terms.append(f\"{coef}*x**{power}\")
    return \" + \".join(terms)

def find_local_extrema(poly):
    # poly is np.poly1d
    d = np.polyder(poly)
    dd = np.polyder(poly, 2)
    # find roots of derivative
    roots = np.roots(d)
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-6]
    extrema = []
    for r in real_roots:
        val = poly(r)
        conc = np.polyval(dd, r)
        kind = 'min' if conc>0 else 'max' if conc<0 else 'inflection'
        extrema.append({'x': r, 'y': val, 'type': kind})
    return extrema

def derivative(poly, x):
    d = np.polyder(poly)
    return np.polyval(d, x)

def domain_range_from_data(x_vals, poly, extrapolate_years=0):
    xmin = min(x_vals)
    xmax = max(x_vals) + extrapolate_years
    # sample to estimate range
    xs = np.linspace(xmin, xmax, 500)
    ys = poly(xs)
    return (xmin, xmax), (float(np.nanmin(ys)), float(np.nanmax(ys)))

def create_printer_html(title, fig_bytes, table_html, analysis_html):
    img_b64 = base64.b64encode(fig_bytes).decode('utf-8')
    html = f\"\"\"
    <html>
    <head>
    <title>{title}</title>
    <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; }}
    h1 {{ font-size: 26px; }}
    .section {{ margin-bottom: 20px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    table, th, td {{ border: 1px solid #666; padding: 6px; }}
    </style>
    </head>
    <body>
    <h1>{title}</h1>
    <div class='section'>
    <h2>Chart</h2>
    <img src='data:image/png;base64,{img_b64}' style='max-width:100%; height:auto;' />
    </div>
    <div class='section'>
    <h2>Raw Data</h2>
    {table_html}
    </div>
    <div class='section'>
    <h2>Analysis</h2>
    {analysis_html}
    </div>
    </body>
    </html>
    \"\"\"
    return html

# --- UI ---
st.title(\"Latin Countries Historical Regression Explorer\")
st.markdown(\"This app fetches historical data (last ~70 years) for several Latin American countries and fits a polynomial regression (degree >=3) to analyze trends, make extrapolations, and perform function analysis.\")


col1, col2 = st.columns([1,2])
with col1:
    category = st.selectbox(\"Choose a data category:\", list(CATEGORIES.keys()))
    countries = st.multiselect(\"Select country/countries to display (Top Latin countries):\", list(TOP_LATIN.keys()), default=[\"Panama\"])
    degree = st.slider(\"Polynomial degree (>=3):\", min_value=3, max_value=8, value=3)
    start_year = st.number_input(\"Start year (>=1950):\", min_value=1900, max_value=CURRENT_YEAR, value=DEFAULT_START_YEAR)
    end_year = st.number_input(\"End year (<=current year):\", min_value=1950, max_value=CURRENT_YEAR, value=CURRENT_YEAR)
    include_extrap = st.checkbox(\"Enable extrapolation into the future?\", value=True)
    extrap_years = st.number_input(\"Extrapolate how many years beyond end year?\", min_value=0, max_value=100, value=10)
    step = st.slider(\"Plot regression evaluation step (years per point)\", min_value=1, max_value=10, value=1)
    show_multiple = st.checkbox(\"Show multiple countries on same graph (comparison)\", value=True)
    compare_us_groups = st.checkbox(\"Compare Latin groups living in the U.S. (sample/optional)\", value=False)
    fit_button = st.button(\"Fetch data & Fit model\")


with col2:
    st.write(\"**Notes & Sources**: Data fetched live from the World Bank API and Our World in Data when available.\")


if fit_button:
    st.info(\"Fetching data...\")
    indicator = CATEGORIES[category]
    # fetch data
    df = fetch_multi(countries, indicator, int(start_year), int(end_year))
    # For education if empty, try OWID fetch for each country
    if indicator == \"MEAN.YEARS.SCHOOLING\" and df.isnull().all().all():
        # try to fetch per country via wb_fetch wrapper
        df = pd.DataFrame(index=range(int(start_year), int(end_year)+1))
        for c in countries:
            iso = TOP_LATIN.get(c)
            s = wb_fetch(iso, indicator, int(start_year), int(end_year))
            df[c] = s
    st.success(\"Data fetched.\")
    # show editable table
    st.subheader(\"Raw data (editable)\")
    df_display = df.copy()
    df_display.index.name = 'Year'
    edited = st.data_editor(df_display.reset_index(), num_rows='dynamic', use_container_width=True)
    # convert edited back to DataFrame with Year index
    edited_df = edited.set_index('Year')
    # choose which country's model to show controls for if multiple selected
    target_country = countries[0] if countries else None
    if show_multiple and len(countries) > 1:
        st.subheader(\"Multiple countries fit on same chart\")
    # plotting area
    chart_col, analysis_col = st.columns([2,1])
    with chart_col:
        fig, ax = plt.subplots(figsize=(10,5))
        xsample = None
        legend_entries = []
        models = {}
        for c in (countries if show_multiple else [target_country]):
            y = edited_df[c].astype(float)
            x = np.array(edited_df.index.astype(int))
            # fit polynomial
            try:
                p, coeffs = poly_fit_years(x.values, y.values, degree=degree)
            except Exception as e:
                st.error(f\"Could not fit polynomial for {c}: {e}\")
                continue
            models[c] = (p, coeffs)
            # plot scatter of available points
            ax.scatter(x, y, label=f\"{c} data\", alpha=0.6)
            # domain for plotting
            xmin = int(x.min())
            xmax = int(x.max()) + (extrap_years if include_extrap else 0)
            xs = np.arange(xmin, xmax+1, step)
            ys = p(xs)
            # For extrapolated region, split into historical and extrapolated for different colors
            if include_extrap:
                mask_hist = xs <= int(end_year)
                ax.plot(xs[mask_hist], ys[mask_hist], label=f\"{c} fit (historical)\")
                if xs[~mask_hist].size>0:
                    ax.plot(xs[~mask_hist], ys[~mask_hist], linestyle='--', label=f\"{c} extrapolation\", alpha=0.75)
            else:
                ax.plot(xs, ys, label=f\"{c} fit\")
            legend_entries.append(c)
            xsample = xs if xsample is None else xsample
        ax.set_xlabel('Year')
        ax.set_ylabel(category)
        ax.set_title(f\"{category} - Polynomial degree {degree}\")
        ax.legend()
        st.pyplot(fig)
        # Save figure to bytes for printer export
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        fig_bytes = buf.getvalue()
    with analysis_col:
        st.subheader(\"Model equations & analysis\")
        analysis_text = \"\"
        for c,(p,coeffs) in models.items():
            eq = poly_to_equation(coeffs)
            st.markdown(f\"**{c} equation:**  $y = {eq}$\")
            # Local extrema
            extrema = find_local_extrema(np.poly1d(coeffs))
            if extrema:
                for e in extrema:
                    # round year to nearest integer for readability
                    year_text = f\"{int(round(e['x']))}\"
                    st.markdown(f\"The {category.lower()} of **{c}** reached a local {e['type']} around {year_text}. The modeled value was approximately {float(round(e['y'],4))}.\")
            else:
                st.markdown(f\"No clear local extrema found for {c} in the polynomial model.\")
            # intervals increasing/decreasing - approximate by sampling derivative sign
            xmin = int(start_year)
            xmax = int(end_year) + (extrap_years if include_extrap else 0)
            xsamp = np.linspace(xmin, xmax, 500)
            dys = derivative(np.poly1d(coeffs), xsamp)
            inc_ranges = []
            dec_ranges = []
            sign = np.sign(dys)
            # find contiguous intervals where sign>0
            def contiguous_intervals(xs, mask):
                intervals = []
                if not mask.any():
                    return intervals
                idx = np.where(mask)[0]
                start = idx[0]
                prev = idx[0]
                for i in idx[1:]:
                    if i == prev+1:
                        prev = i
                    else:
                        intervals.append((xs[start], xs[prev]))
                        start = i; prev = i
                intervals.append((xs[start], xs[prev]))
                return intervals
            inc_intervals = contiguous_intervals(xsamp, sign>0)
            dec_intervals = contiguous_intervals(xsamp, sign<0)
            if inc_intervals:
                txt = \"; \".join([f\"{int(round(a))} to {int(round(b))}\" for a,b in inc_intervals])
                st.markdown(f\"{c} is modeled as increasing on intervals (approx): {txt}.\")
            if dec_intervals:
                txt = \"; \".join([f\"{int(round(a))} to {int(round(b))}\" for a,b in dec_intervals])
                st.markdown(f\"{c} is modeled as decreasing on intervals (approx): {txt}.\")
            # fastest increase/decrease -> max/min of derivative magnitude
            idx_max = np.argmax(dys)
            idx_min = np.argmin(dys)
            fastest_inc_year = int(round(xsamp[idx_max]))
            fastest_dec_year = int(round(xsamp[idx_min]))
            st.markdown(f\"According to the model, {c} was increasing fastest around {fastest_inc_year} (derivative approx {dys[idx_max]:.4f}) and decreasing fastest around {fastest_dec_year} (derivative approx {dys[idx_min]:.4f}).\")
            # domain & range
            dom, rng = domain_range_from_data(edited_df.index.astype(int), np.poly1d(coeffs), extrapolate_years=extrap_years if include_extrap else 0)
            st.markdown(f\"Domain (years considered): {int(dom[0])} to {int(dom[1])}. Estimated range (model sample): {rng[0]:.4f} to {rng[1]:.4f}.\")
            # Conjecture section (very basic heuristic)
            st.markdown(f\"**Conjectures for significant changes in {c}:** Consider historical events, economic crises, political changes, wars, and policy reforms. For example, for {c}, compare dates of rapid model change to known events (economic crises, political transitions, pandemics) to hypothesize causes.\") 
            analysis_text += f\"\\n\\n{c}: equation: {eq}\\nExtrema: {extrema}\\n\"
        # Extrapolation prediction example (predict value in future year)
        future_year = int(end_year) + int(extrap_years)
        for c,(p,coeffs) in models.items():
            pred = float(np.polyval(np.poly1d(coeffs), future_year))
            st.markdown(f\"**Extrapolation:** According to the model for **{c}**, the {category.lower()} in year {future_year} is predicted to be approximately **{pred:.4f}**.\")
    # Additional interactive tools
    st.subheader(\"Interpolation / Extrapolation calculator\")
    calc_country = st.selectbox(\"Choose country for calculator:\", countries)
    calc_year = st.number_input(\"Year to evaluate (can be outside range):\", value=int(end_year)+1)
    if calc_country in models:
        p,coeffs = models[calc_country]
        val = float(np.polyval(p, calc_year))
        st.write(f\"Model prediction for {calc_country} in {calc_year}: {val:.6f} {''}\")
    st.subheader(\"Average rate of change between two years (model-based)\")
    a_year = st.number_input(\"First year:\", value=int(start_year), key='a_year')
    b_year = st.number_input(\"Second year:\", value=int(start_year)+5, key='b_year')
    roc_country = st.selectbox(\"Country for rate of change:\", countries, key='roc_country')
    if roc_country in models and b_year != a_year:
        p,_ = models[roc_country]
        f_a = float(np.polyval(p, a_year))
        f_b = float(np.polyval(p, b_year))
        avg_roc = (f_b - f_a)/(b_year - a_year)
        st.write(f\"Average rate of change for {roc_country} from {a_year} to {b_year}: {avg_roc:.6f} per year.\")
    # Download printer-friendly HTML
    st.subheader(\"Printer-friendly export\")
    table_html = edited_df.to_html()
    analysis_html = \"<pre>\" + analysis_text + \"</pre>\"
    html = create_printer_html(f\"{category} analysis\", fig_bytes, table_html, analysis_html)
    b64 = base64.b64encode(html.encode('utf-8')).decode('utf-8')
    href = f\"data:text/html;base64,{b64}\"
    st.markdown(f\"[Download printer-friendly HTML]({href})\", unsafe_allow_html=True)

    # Also allow CSV download of edited table
    csv = edited_df.to_csv().encode('utf-8')
    st.download_button(\"Download data (CSV)\", csv, file_name=f\"{category.replace(' ','_')}_data.csv\", mime='text/csv')

    st.success(\"Analysis complete.\")


st.markdown(\"---\")
st.caption(\"Built to fetch World Bank & Our World in Data series. The world bank provides many historical series back to 1960; some indicators may not have full coverage to 70 years.\")

