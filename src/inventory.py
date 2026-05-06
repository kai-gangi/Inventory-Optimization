from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InventoryPolicy:
    lead_time_days: int
    service_z: float
    review_period_days: int
    min_order_qty: float = 0.0


def recommended_policy(train_demand: pd.Series) -> InventoryPolicy:
    mean = float(train_demand.mean())

    if mean < 200:
        return InventoryPolicy(lead_time_days=7, service_z=1.65, review_period_days=7)
    if mean < 260:
        return InventoryPolicy(lead_time_days=10, service_z=1.85, review_period_days=7)
    return InventoryPolicy(lead_time_days=14, service_z=2.05, review_period_days=7)


def reorder_point(mean_daily_demand: float, std_daily_demand: float, lead_time_days: int, z: float) -> float:
    lead_time_demand = mean_daily_demand * lead_time_days
    lead_time_std = std_daily_demand * np.sqrt(lead_time_days)
    return max(0.0, lead_time_demand + z * lead_time_std)


def order_up_to_target(
    mean_daily_demand: float,
    std_daily_demand: float,
    lead_time_days: int,
    review_period_days: int,
    z: float,
) -> float:
    protection_period = lead_time_days + review_period_days
    protection_mean = mean_daily_demand * protection_period
    protection_std = std_daily_demand * np.sqrt(protection_period)
    return max(0.0, protection_mean + z * protection_std)


def simulate_inventory(
    actual_demand: np.ndarray,
    forecast_demand: np.ndarray,
    policy: InventoryPolicy,
    initial_inventory: float,
) -> pd.DataFrame:
    n = len(actual_demand)
    receipts = np.zeros(n)
    inventory = float(initial_inventory)
    backlog = 0.0

    in_transit: list[tuple[int, float]] = []
    rows = []

    for t in range(n):
        arriving = sum(qty for arrival_day, qty in in_transit if arrival_day == t)
        if arriving > 0:
            inventory += arriving
        in_transit = [(arrival_day, qty) for arrival_day, qty in in_transit if arrival_day != t]

        demand_t = float(actual_demand[t] + backlog)
        fulfilled = min(inventory, demand_t)
        inventory -= fulfilled
        backlog = demand_t - fulfilled

        if t % policy.review_period_days == 0:
            lookback_start = max(0, t - 30)
            hist_forecast = forecast_demand[lookback_start : t + 1]
            mu = float(np.mean(hist_forecast)) if len(hist_forecast) else float(forecast_demand[t])
            sigma = float(np.std(hist_forecast)) if len(hist_forecast) else 0.0

            rop = reorder_point(mu, sigma, policy.lead_time_days, policy.service_z)
            target = order_up_to_target(mu, sigma, policy.lead_time_days, policy.review_period_days, policy.service_z)

            on_order = sum(qty for _, qty in in_transit)
            inventory_position = inventory + on_order - backlog

            if inventory_position < rop:
                order_qty = max(policy.min_order_qty, target - inventory_position)
                if order_qty > 0:
                    arrival_day = min(n - 1, t + policy.lead_time_days)
                    in_transit.append((arrival_day, order_qty))
                    receipts[arrival_day] += order_qty

        rows.append(
            {
                "t": t + 1,
                "actual_demand": float(actual_demand[t]),
                "demand_with_backlog": demand_t,
                "fulfilled": fulfilled,
                "inventory_end": inventory,
                "backlog_end": backlog,
                "on_order_qty": sum(qty for _, qty in in_transit),
            }
        )

    return pd.DataFrame(rows)


def inventory_kpis(df: pd.DataFrame) -> dict:
    total_need = float(df["demand_with_backlog"].sum())
    total_fulfilled = float(df["fulfilled"].sum())

    return {
        "service_level": (total_fulfilled / total_need) if total_need else 0.0,
        "avg_inventory": float(df["inventory_end"].mean()),
        "avg_backlog": float(df["backlog_end"].mean()),
        "max_backlog": float(df["backlog_end"].max()),
    }
