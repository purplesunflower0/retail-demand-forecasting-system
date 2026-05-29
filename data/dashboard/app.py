import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Retail Demand Forecasting Dashboard",
    layout="wide"
)

@st.cache_data
def load_data():
    forecast = pd.read_csv("forecast_vs_actual.csv", parse_dates=["date"])
    segment_metrics = pd.read_csv("segment_metrics.csv")
    product_metrics = pd.read_csv("product_metrics.csv")
    reliability = pd.read_csv("reliability_metrics.csv")
    return forecast, segment_metrics, product_metrics, reliability

forecast_df, segment_metrics_df, product_metrics_df, reliability_df = load_data()

st.title("📦 Retail Demand Forecasting System")
st.caption(
    "Segment-aware forecasting with baseline, Prophet, and LightGBM models"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🧠 Model Routing",
    "📈 Product Drilldown",
    "🛡 Reliability & Trust"
])


with tab1:
    st.subheader("System-Level Performance")
    overall_mae = forecast_df["prediction"].sub(forecast_df["sales"]).abs().mean()
    overall_rmse = np.sqrt(
        ((forecast_df["prediction"] - forecast_df["sales"]) ** 2).mean()
    )
    overall_bias = (forecast_df["prediction"] - forecast_df["sales"]).mean()

    reliability_pct = reliability_df["pct_within_1"].mean() * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Overall MAE", f"{overall_mae:.2f}")
    c2.metric("Overall RMSE", f"{overall_rmse:.2f}")
    c3.metric("Forecast Bias", f"{overall_bias:.2f}")
    c4.metric("Reliability (%)", f"{reliability_pct:.1f}%")


with tab2:
    st.subheader("Model Routing by Product Segment")
    segment_counts = (
        product_metrics_df[["item_id", "segment"]]
        .drop_duplicates()
        .value_counts("segment")
        .reset_index(name="num_products")
    )

    fig_seg = px.bar(
        segment_counts,
        x="segment",
        y="num_products",
        title="Products per Demand Segment"
    )

    st.plotly_chart(fig_seg, use_container_width=True)
    fig_perf = px.bar(
        segment_metrics_df,
        x="segment",
        y="mae",
        color="model_used",
        barmode="group",
        title="MAE by Segment and Model"
    )

    st.plotly_chart(fig_perf, use_container_width=True)


with tab3:
    st.subheader("Product-Level Forecast Inspection")
    item_list = sorted(
        forecast_df["item_id"]
        .dropna()
        .astype(str)
        .unique()
    )
    selected_item = st.selectbox("Select Product", item_list)
    item_df = forecast_df[forecast_df["item_id"] == selected_item]
    item_meta = product_metrics_df[product_metrics_df["item_id"] == selected_item].iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Segment", item_meta["segment"])
    c2.metric("Model Used", item_meta["model_used"])
    c3.metric("Product MAE", f"{item_meta['mae']:.2f}")
    c4.metric("Product Bias", f"{item_meta['bias']:.2f}")
    fig_ts = px.line(
        item_df,
        x="date",
        y=["sales", "prediction"],
        labels={"value": "Units Sold", "variable": "Series"},
        title=f"Actual vs Forecast — {selected_item}"
    )

    st.plotly_chart(fig_ts, use_container_width=True)

with tab4:
    st.subheader("Forecast Reliability")
    fig_rel = px.bar(
        reliability_df,
        x="model_used",
        y="pct_within_1",
        title="Reliability: % Predictions Within ±1 Unit",
        labels={"pct_within_1": "Reliability"}
    )

    st.plotly_chart(fig_rel, use_container_width=True)


    forecast_df["abs_error"] = (
        forecast_df["prediction"] - forecast_df["sales"]
    ).abs()

    fig_err = px.histogram(
        forecast_df,
        x="abs_error",
        nbins=50,
        title="Absolute Error Distribution"
    )

    st.plotly_chart(fig_err, use_container_width=True)
