"""Streamlit dashboard: demand forecast + customer segmentation, side by side.

Run: streamlit run app.py
"""
import os
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from forecast import run_forecast
from segmentation import df as seg_df  # runs segmentation on import (cheap: 20k rows)

if not os.path.exists("data/daily_sales.csv") or not os.path.exists("data/customers.csv"):
    import subprocess
    subprocess.run(["python", "data/generate_data.py"], check=True)


def main():
    st.set_page_config(page_title="Meesho-style Demand & Segmentation Dashboard", page_icon="📈", layout="wide")
    st.title("📈 Tier 2/3 Demand Forecast & Customer Segmentation")

    tab1, tab2 = st.tabs(["Demand Forecast", "Customer Segmentation"])

    with tab1:
        st.caption("Forecasts daily order volume per city-tier x category, with Indian "
                   "festive-season spikes modeled explicitly (Republic Day, Diwali, etc.)")
        col1, col2, col3 = st.columns(3)
        city_tier = col1.selectbox("City Tier", ["Tier1", "Tier2", "Tier3"], index=1)
        category = col2.selectbox("Category", ["Sarees", "Kurtis", "Western Wear", "Jewellery",
                                                "Kids Wear", "Footwear", "Home Furnishing", "Men's Wear"])
        horizon = col3.slider("Forecast horizon (days)", 14, 120, 60)

        if st.button("Run forecast"):
            with st.spinner("Fitting Prophet model..."):
                result = run_forecast(city_tier, category, horizon, plot=False)
            st.line_chart(result.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]])
            st.dataframe(result.tail(10), width="stretch")

    with tab2:
        st.caption("RFM (Recency/Frequency/Monetary) segments via KMeans — "
                   "Tier2/3 customers skew toward lower monetary but far higher volume.")
        st.dataframe(seg_df["segment_label"].value_counts().rename("count"))
        fig, ax = plt.subplots(figsize=(7, 5))
        for seg in seg_df["segment_label"].unique():
            sub = seg_df[seg_df["segment_label"] == seg]
            ax.scatter(sub["recency_days"], sub["monetary"], s=8, alpha=0.4, label=seg)
        ax.set_xlabel("Recency (days)"); ax.set_ylabel("Monetary (₹)"); ax.legend()
        st.pyplot(fig)

        st.subheader("Segment mix by city tier")
        st.dataframe(pd.crosstab(seg_df["city_tier"], seg_df["segment_label"]), width="stretch")


try:
    import streamlit.web.bootstrap as bootstrap
    bootstrap.run(main, args=[], flag_options={})
except Exception:
    main()