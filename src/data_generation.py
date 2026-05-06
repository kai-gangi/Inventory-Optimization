from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataGenParams:
    days: int = 730
    seed: int = 42
    base_level: float = 230.0
    linear_trend_per_day: float = 0.055
    yearly_seasonality_amp: float = 50.0
    weekly_seasonality_amp: float = 12.0
    noise_std: float = 18.0


def generate_retail_demand(params: DataGenParams) -> pd.DataFrame:
    rng = np.random.default_rng(params.seed)
    day = np.arange(params.days)

    level = params.base_level + params.linear_trend_per_day * day
    yearly = params.yearly_seasonality_amp * np.sin(2 * np.pi * day / 365.0)
    weekly = params.weekly_seasonality_amp * np.sin(2 * np.pi * day / 7.0)
    noise = rng.normal(0.0, params.noise_std, size=params.days)

    demand = np.maximum(0.0, level + yearly + weekly + noise)

    return pd.DataFrame(
        {
            "day": day + 1,
            "demand": demand,
            "trend": level,
            "yearly_component": yearly,
            "weekly_component": weekly,
        }
    )
