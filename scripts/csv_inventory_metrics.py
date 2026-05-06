from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting import ForecastingParams, evaluate_models
from src.inventory import InventoryPolicy, inventory_kpis, recommended_policy, simulate_inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inventory optimization metrics on a CSV demand column."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="data/retail_store_inventory.csv",
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--demand-col",
        type=str,
        default="Demand Forecast",
        help="Column to treat as demand.",
    )
    parser.add_argument(
        "--date-col",
        type=str,
        default="Date",
        help="Date column used for daily aggregation.",
    )
    parser.add_argument(
        "--aggregate",
        choices=["sum", "mean"],
        default="sum",
        help="How to aggregate demand column per day.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--moving-avg-window", type=int, default=14)
    parser.add_argument("--initial-inventory", type=float, default=1400.0)
    parser.add_argument("--lead-time", type=int, default=10)
    parser.add_argument("--service-z", type=float, default=1.85)
    parser.add_argument("--review-period", type=int, default=7)
    return parser.parse_args()


def pct_change(new_val: float, old_val: float) -> float:
    if old_val == 0:
        return 0.0
    return (new_val - old_val) / old_val


def load_daily_demand(csv_path: Path, demand_col: str, date_col: str, aggregate: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if demand_col not in df.columns:
        raise ValueError(f"Demand column '{demand_col}' not found. Columns: {list(df.columns)}")
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found. Columns: {list(df.columns)}")

    df = df[[date_col, demand_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[demand_col] = pd.to_numeric(df[demand_col], errors="coerce")
    df = df.dropna(subset=[date_col, demand_col])

    daily = (
        df.groupby(date_col, as_index=False)[demand_col].sum()
        if aggregate == "sum"
        else df.groupby(date_col, as_index=False)[demand_col].mean()
    )
    daily = daily.sort_values(date_col).reset_index(drop=True)
    daily = daily.rename(columns={demand_col: "demand"})
    daily["day"] = daily.index + 1

    return daily[["day", "demand"]]


def main() -> None:
    args = parse_args()

    csv_path = Path(args.input_csv)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

    demand_df = load_daily_demand(
        csv_path=csv_path,
        demand_col=args.demand_col,
        date_col=args.date_col,
        aggregate=args.aggregate,
    )

    forecast_params = ForecastingParams(
        train_ratio=args.train_ratio,
        moving_avg_window=args.moving_avg_window,
    )
    forecast_output = evaluate_models(demand_df, forecast_params)

    test = forecast_output["test"].copy()
    y_actual = test["demand"].to_numpy()

    y_baseline = forecast_output["predictions"]["naive"]
    y_best = forecast_output["predictions"][forecast_output["best_model"]]

    baseline_policy = recommended_policy(forecast_output["train"]["demand"])
    optimized_policy = InventoryPolicy(
        lead_time_days=args.lead_time,
        service_z=args.service_z,
        review_period_days=args.review_period,
    )

    baseline_inv = simulate_inventory(
        actual_demand=y_actual,
        forecast_demand=y_baseline,
        policy=baseline_policy,
        initial_inventory=args.initial_inventory,
    )
    optimized_inv = simulate_inventory(
        actual_demand=y_actual,
        forecast_demand=y_best,
        policy=optimized_policy,
        initial_inventory=args.initial_inventory,
    )

    baseline_kpi = inventory_kpis(baseline_inv)
    optimized_kpi = inventory_kpis(optimized_inv)

    baseline_mae = forecast_output["scores"]["naive"]
    best_mae = forecast_output["scores"][forecast_output["best_model"]]

    print("=== INPUT SUMMARY ===")
    print(f"CSV: {csv_path}")
    print(f"Demand column: {args.demand_col}")
    print(f"Date column: {args.date_col}")
    print(f"Aggregation: {args.aggregate}")
    print(f"Rows after daily aggregation: {len(demand_df)}")

    print("\n=== FORECAST METRICS (MAE) ===")
    for model_name, model_mae in forecast_output["scores"].items():
        print(f"{model_name:28s}: {model_mae:.2f}")
    print(f"Best model: {forecast_output['best_model']}")
    print(f"MAE improvement vs naive: {((baseline_mae - best_mae) / baseline_mae):.2%}")

    print("\n=== INVENTORY KPI COMPARISON ===")
    print(f"Baseline service level:  {baseline_kpi['service_level']:.2%}")
    print(f"Optimized service level: {optimized_kpi['service_level']:.2%}")
    print(f"Service level delta:     {optimized_kpi['service_level'] - baseline_kpi['service_level']:.2%}")

    print(f"\nBaseline avg backlog:    {baseline_kpi['avg_backlog']:.2f}")
    print(f"Optimized avg backlog:   {optimized_kpi['avg_backlog']:.2f}")
    print(f"Avg backlog % change:    {pct_change(optimized_kpi['avg_backlog'], baseline_kpi['avg_backlog']):.2%}")

    print(f"\nBaseline max backlog:    {baseline_kpi['max_backlog']:.2f}")
    print(f"Optimized max backlog:   {optimized_kpi['max_backlog']:.2f}")
    print(f"Max backlog % change:    {pct_change(optimized_kpi['max_backlog'], baseline_kpi['max_backlog']):.2%}")


if __name__ == "__main__":
    main()
