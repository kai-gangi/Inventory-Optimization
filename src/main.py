from __future__ import annotations

import argparse
from pathlib import Path

from .data_generation import DataGenParams, generate_retail_demand
from .data_io import load_demand_from_csv
from .forecasting import ForecastingParams, evaluate_models
from .inventory import InventoryPolicy, inventory_kpis, recommended_policy, simulate_inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demand Forecasting & Inventory Optimization Engine")

    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--moving-avg-window", type=int, default=14)
    parser.add_argument("--initial-inventory", type=float, default=1400.0)

    parser.add_argument("--lead-time", type=int, default=10)
    parser.add_argument("--service-z", type=float, default=1.85)
    parser.add_argument("--review-period", type=int, default=7)

    parser.add_argument("--output-eval", type=str, default="")
    parser.add_argument("--output-inventory", type=str, default="")

    parser.add_argument("--input-csv", type=str, default="")
    parser.add_argument("--input-demand-col", type=str, default="")
    parser.add_argument("--input-date-col", type=str, default="")

    return parser.parse_args()


def pct_change(new_val: float, old_val: float) -> float:
    if old_val == 0:
        return 0.0
    return (new_val - old_val) / old_val


def main() -> None:
    args = parse_args()

    if args.input_csv:
        demand_df = load_demand_from_csv(
            args.input_csv,
            demand_col=(args.input_demand_col or None),
            date_col=(args.input_date_col or None),
        )
    else:
        demand_params = DataGenParams(days=args.days, seed=args.seed)
        demand_df = generate_retail_demand(demand_params)

    forecast_params = ForecastingParams(train_ratio=args.train_ratio, moving_avg_window=args.moving_avg_window)
    forecast_output = evaluate_models(demand_df, forecast_params)

    test = forecast_output["test"].copy()
    y_actual = test["demand"].to_numpy()

    y_baseline = forecast_output["predictions"]["naive"]
    y_best = forecast_output["predictions"][forecast_output["best_model"]]

    print("\n=== Forecast Model Evaluation (MAE) ===")
    for model_name, model_mae in forecast_output["scores"].items():
        print(f"{model_name:28s}: {model_mae:.2f}")

    baseline_mae = forecast_output["scores"]["naive"]
    best_mae = forecast_output["scores"][forecast_output["best_model"]]

    print(f"\nBest model: {forecast_output['best_model']}")
    print(f"MAE improvement vs naive: {((baseline_mae - best_mae) / baseline_mae):.2%}")

    auto_policy = recommended_policy(forecast_output["train"]["demand"])
    scenario_policy = InventoryPolicy(
        lead_time_days=args.lead_time,
        service_z=args.service_z,
        review_period_days=args.review_period,
    )

    baseline_inv = simulate_inventory(
        actual_demand=y_actual,
        forecast_demand=y_baseline,
        policy=auto_policy,
        initial_inventory=args.initial_inventory,
    )
    optimized_inv = simulate_inventory(
        actual_demand=y_actual,
        forecast_demand=y_best,
        policy=scenario_policy,
        initial_inventory=args.initial_inventory,
    )

    baseline_kpis = inventory_kpis(baseline_inv)
    optimized_kpis = inventory_kpis(optimized_inv)

    print("\n=== Inventory Policy Results ===")
    print(f"Baseline service level:  {baseline_kpis['service_level']:.2%}")
    print(f"Optimized service level: {optimized_kpis['service_level']:.2%}")
    print(f"Service level delta:     {optimized_kpis['service_level'] - baseline_kpis['service_level']:.2%}")

    print(f"\nBaseline avg backlog:    {baseline_kpis['avg_backlog']:.2f}")
    print(f"Optimized avg backlog:   {optimized_kpis['avg_backlog']:.2f}")
    print(f"Avg backlog % change:    {pct_change(optimized_kpis['avg_backlog'], baseline_kpis['avg_backlog']):.2%}")

    print(f"\nBaseline max backlog:    {baseline_kpis['max_backlog']:.2f}")
    print(f"Optimized max backlog:   {optimized_kpis['max_backlog']:.2f}")
    print(f"Max backlog % change:    {pct_change(optimized_kpis['max_backlog'], baseline_kpis['max_backlog']):.2%}")

    eval_frame = forecast_output["evaluation_frame"]
    inventory_compare = baseline_inv.copy()
    inventory_compare = inventory_compare.rename(
        columns={
            "fulfilled": "fulfilled_baseline",
            "inventory_end": "inventory_end_baseline",
            "backlog_end": "backlog_end_baseline",
            "on_order_qty": "on_order_qty_baseline",
        }
    )
    inventory_compare["fulfilled_optimized"] = optimized_inv["fulfilled"]
    inventory_compare["inventory_end_optimized"] = optimized_inv["inventory_end"]
    inventory_compare["backlog_end_optimized"] = optimized_inv["backlog_end"]
    inventory_compare["on_order_qty_optimized"] = optimized_inv["on_order_qty"]

    if args.output_eval:
        out_eval = Path(args.output_eval)
        out_eval.parent.mkdir(parents=True, exist_ok=True)
        eval_frame.to_csv(out_eval, index=False)
        print(f"\nSaved forecast evaluation output: {out_eval}")

    if args.output_inventory:
        out_inventory = Path(args.output_inventory)
        out_inventory.parent.mkdir(parents=True, exist_ok=True)
        inventory_compare.to_csv(out_inventory, index=False)
        print(f"Saved inventory simulation output: {out_inventory}")


if __name__ == "__main__":
    main()
