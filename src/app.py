from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generation import DataGenParams, generate_retail_demand
from src.data_io import load_demand_from_csv
from src.forecasting import ForecastingParams, evaluate_models
from src.inventory import InventoryPolicy, inventory_kpis, recommended_policy, simulate_inventory

st.set_page_config(page_title="Demand & Inventory Studio", page_icon="📈", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}

.block-container {
    padding-top: 1.4rem;
}

.kpi-card {
    border: 1px solid #d7e1ea;
    border-radius: 14px;
    padding: 0.85rem 1rem;
    background: linear-gradient(145deg, #f8fbff 0%, #edf5ff 100%);
}

.kpi-label {
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 0.76rem;
    color: #335b84;
}

.kpi-value {
    font-size: 1.45rem;
    color: #0f2740;
    font-weight: 800;
}

.mono {
    font-family: 'IBM Plex Mono', monospace;
    color: #445b72;
}
</style>
""",
    unsafe_allow_html=True,
)


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


st.title("📈 Demand Forecasting & Inventory Optimization Studio")
st.markdown('<div class="mono">Forecasting + replenishment policy simulation for retail demand</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Experiment Settings")

    days = st.slider("Days", min_value=365, max_value=1460, value=730, step=35)
    seed = st.number_input("Seed", min_value=0, max_value=99999, value=42, step=1)

    st.subheader("Upload demand CSV")
    uploaded_file = st.file_uploader("Upload demand CSV", type=["csv"])
    input_demand_col = st.text_input("Demand column name", value="")
    input_date_col = st.text_input("Date column name (optional)", value="")

    st.subheader("Demand Generator")
    base_level = st.slider("Base Demand Level", min_value=100, max_value=500, value=230, step=5)
    trend = st.slider("Linear Trend / Day", min_value=0.0, max_value=0.20, value=0.055, step=0.005)
    yearly_amp = st.slider("Yearly Seasonality", min_value=0, max_value=120, value=50, step=5)
    weekly_amp = st.slider("Weekly Seasonality", min_value=0, max_value=40, value=12, step=1)
    noise_std = st.slider("Noise Std Dev", min_value=1, max_value=60, value=18, step=1)

    st.subheader("Forecasting")
    train_ratio = st.slider("Train Ratio", min_value=0.6, max_value=0.9, value=0.8, step=0.05)
    ma_window = st.slider("Moving Avg Window", min_value=3, max_value=60, value=14, step=1)

    st.subheader("Optimized Inventory Policy")
    initial_inventory = st.slider("Initial Inventory", min_value=100, max_value=5000, value=1400, step=50)
    lead_time = st.slider("Lead Time (days)", min_value=1, max_value=30, value=10, step=1)
    service_z = st.slider("Service Z", min_value=0.5, max_value=3.0, value=1.85, step=0.05)
    review_period = st.slider("Review Period (days)", min_value=1, max_value=30, value=7, step=1)

    run_button = st.button("Run Analysis", type="primary", use_container_width=True)


if run_button:
    if uploaded_file is not None:
        demand_df = load_demand_from_csv(uploaded_file, demand_col=(input_demand_col or None), date_col=(input_date_col or None))
    else:
        demand_params = DataGenParams(
            days=days,
            seed=seed,
            base_level=float(base_level),
            linear_trend_per_day=float(trend),
            yearly_seasonality_amp=float(yearly_amp),
            weekly_seasonality_amp=float(weekly_amp),
            noise_std=float(noise_std),
        )
        demand_df = generate_retail_demand(demand_params)

    forecast_params = ForecastingParams(train_ratio=float(train_ratio), moving_avg_window=int(ma_window))
    forecast_output = evaluate_models(demand_df, forecast_params)

    test_df = forecast_output["test"].copy()
    y_actual = test_df["demand"].to_numpy()
    y_baseline = forecast_output["predictions"]["naive"]
    y_best = forecast_output["predictions"][forecast_output["best_model"]]

    baseline_policy = recommended_policy(forecast_output["train"]["demand"])
    optimized_policy = InventoryPolicy(
        lead_time_days=int(lead_time),
        service_z=float(service_z),
        review_period_days=int(review_period),
    )

    baseline_inv = simulate_inventory(y_actual, y_baseline, baseline_policy, float(initial_inventory))
    optimized_inv = simulate_inventory(y_actual, y_best, optimized_policy, float(initial_inventory))

    baseline_kpi = inventory_kpis(baseline_inv)
    optimized_kpi = inventory_kpis(optimized_inv)

    score_naive = float(forecast_output["scores"]["naive"])
    score_ma = float(forecast_output["scores"]["moving_average"])
    score_reg = float(forecast_output["scores"]["trend_seasonality_regression"])
    score_best = float(forecast_output["scores"][forecast_output["best_model"]])

    st.subheader("Forecasting Performance")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Naive MAE", f"{score_naive:.2f}")
    with c2:
        metric_card("Moving Avg MAE", f"{score_ma:.2f}")
    with c3:
        metric_card("Regression MAE", f"{score_reg:.2f}")
    with c4:
        metric_card("MAE Gain vs Naive", f"{((score_naive - score_best) / score_naive):.2%}")

    st.subheader("Inventory Outcomes")
    d1, d2, d3 = st.columns(3)
    d1.metric(
        "Service Level",
        f"{optimized_kpi['service_level']:.2%}",
        f"{optimized_kpi['service_level'] - baseline_kpi['service_level']:.2%}",
    )
    d2.metric(
        "Average Backlog",
        f"{optimized_kpi['avg_backlog']:.2f}",
        f"{optimized_kpi['avg_backlog'] - baseline_kpi['avg_backlog']:.2f}",
        delta_color="inverse",
    )
    d3.metric(
        "Max Backlog",
        f"{optimized_kpi['max_backlog']:.2f}",
        f"{optimized_kpi['max_backlog'] - baseline_kpi['max_backlog']:.2f}",
        delta_color="inverse",
    )

    st.subheader("Demand and Forecast Traces")
    eval_df = forecast_output["evaluation_frame"].copy()
    eval_df = eval_df.rename(
        columns={
            "pred_naive": "naive_forecast",
            "pred_moving_average": "moving_average_forecast",
            "pred_regression": "regression_forecast",
        }
    )
    # If the dataset is large, allow downsampling for display to keep the UI responsive.
    total_days = int(demand_df.shape[0])
    if total_days > 2000:
        st.warning(f"Dataset has {total_days} rows — large datasets may be slow to render. Charts will use a sampled view by default.")
        display_choice = st.selectbox(
            "Display mode",
            [
                "Last 730 days (recommended)",
                "Weekly downsample (avg)",
                "First 500 rows",
                "Full (may be slow)",
            ],
            index=0,
        )
    else:
        display_choice = "Full (may be slow)"

    def make_display_series(df: pd.DataFrame, choice: str) -> pd.DataFrame:
        if choice == "Full (may be slow)":
            return df
        if choice == "Last 730 days (recommended)":
            return df.tail(730).reset_index(drop=True)
        if choice == "First 500 rows":
            return df.head(500).reset_index(drop=True)
        # weekly downsample
        # assume `day` is 1-based contiguous
        df2 = df.copy().reset_index(drop=True)
        df2["wk"] = ((df2["day"] - 1) // 7).astype(int)
        agg_map = {
            "day": "first",
            "demand": "mean",
            "naive_forecast": "mean",
            "moving_average_forecast": "mean",
            "regression_forecast": "mean",
        }
        return df2.groupby("wk").agg(agg_map).reset_index(drop=True)

    display_eval = make_display_series(eval_df, display_choice)

    st.line_chart(
        display_eval.set_index("day")[["demand", "naive_forecast", "moving_average_forecast", "regression_forecast"]],
        color=["#0f2740", "#b45309", "#16a34a", "#2563eb"],
    )

    st.subheader("Backlog Comparison")
    compare_df = pd.DataFrame(
        {
            "t": optimized_inv["t"],
            "baseline_backlog": baseline_inv["backlog_end"],
            "optimized_backlog": optimized_inv["backlog_end"],
            "baseline_inventory": baseline_inv["inventory_end"],
            "optimized_inventory": optimized_inv["inventory_end"],
        }
    )

    left, right = st.columns(2)
    with left:
        st.markdown("Backlog Trajectory")
        st.area_chart(compare_df.set_index("t")[["baseline_backlog", "optimized_backlog"]], color=["#9f1239", "#16a34a"])
    with right:
        st.markdown("Inventory Position")
        st.line_chart(compare_df.set_index("t")[["baseline_inventory", "optimized_inventory"]], color=["#64748b", "#2563eb"])

    st.subheader("Policy Settings")
    p1, p2 = st.columns(2)
    with p1:
        st.json(
            {
                "baseline_policy_auto": {
                    "lead_time_days": baseline_policy.lead_time_days,
                    "service_z": baseline_policy.service_z,
                    "review_period_days": baseline_policy.review_period_days,
                }
            }
        )
    with p2:
        st.json(
            {
                "optimized_policy_user": {
                    "lead_time_days": optimized_policy.lead_time_days,
                    "service_z": optimized_policy.service_z,
                    "review_period_days": optimized_policy.review_period_days,
                }
            }
        )

    st.subheader("Download Outputs")
    b1, b2 = st.columns(2)
    with b1:
        st.download_button(
            "Download Forecast Evaluation CSV",
            data=to_csv_bytes(eval_df),
            file_name="forecast_evaluation.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with b2:
        st.download_button(
            "Download Inventory Comparison CSV",
            data=to_csv_bytes(compare_df),
            file_name="inventory_comparison.csv",
            mime="text/csv",
            use_container_width=True,
        )
else:
    st.info("Configure parameters in the sidebar and click Run Analysis.")
