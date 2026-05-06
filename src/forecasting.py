from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastingParams:
    train_ratio: float = 0.8
    moving_avg_window: int = 14


def split_train_test(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def moving_average_forecast(train: pd.DataFrame, test: pd.DataFrame, window: int) -> np.ndarray:
    history = train["demand"].tolist()
    preds = []

    for actual in test["demand"].tolist():
        recent = history[-window:] if len(history) >= window else history
        pred = float(np.mean(recent)) if recent else 0.0
        preds.append(pred)
        history.append(actual)

    return np.array(preds)


def naive_forecast(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Forecast every test point as the last observed training demand."""
    return np.full(shape=len(test), fill_value=float(train["demand"].iloc[-1]))


def _feature_matrix(day_values: np.ndarray) -> np.ndarray:
    yearly_phase = 2 * np.pi * day_values / 365.0
    weekly_phase = 2 * np.pi * day_values / 7.0

    return np.column_stack(
        [
            np.ones_like(day_values, dtype=float),
            day_values.astype(float),
            np.sin(yearly_phase),
            np.cos(yearly_phase),
            np.sin(weekly_phase),
            np.cos(weekly_phase),
        ]
    )


def trend_seasonality_regression_forecast(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    x_train = _feature_matrix(train["day"].to_numpy())
    y_train = train["demand"].to_numpy()

    beta, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)

    x_test = _feature_matrix(test["day"].to_numpy())
    preds = x_test @ beta
    return np.maximum(0.0, preds)


def evaluate_models(df: pd.DataFrame, params: ForecastingParams) -> Dict[str, object]:
    train, test = split_train_test(df, params.train_ratio)

    y_test = test["demand"].to_numpy()

    y_naive = naive_forecast(train, test)
    y_ma = moving_average_forecast(train, test, params.moving_avg_window)
    y_reg = trend_seasonality_regression_forecast(train, test)

    scores = {
        "naive": mae(y_test, y_naive),
        "moving_average": mae(y_test, y_ma),
        "trend_seasonality_regression": mae(y_test, y_reg),
    }

    best_model = min(scores, key=scores.get)
    predictions = {
        "naive": y_naive,
        "moving_average": y_ma,
        "trend_seasonality_regression": y_reg,
    }

    eval_df = test[["day", "demand"]].copy()
    eval_df["pred_naive"] = y_naive
    eval_df["pred_moving_average"] = y_ma
    eval_df["pred_regression"] = y_reg

    return {
        "train": train,
        "test": test,
        "scores": scores,
        "best_model": best_model,
        "predictions": predictions,
        "evaluation_frame": eval_df,
    }
