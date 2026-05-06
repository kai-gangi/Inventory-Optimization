from pathlib import Path
from typing import Optional, Union, IO

import pandas as pd


def load_demand_from_csv(path_or_buffer: Union[str, Path, IO], demand_col: Optional[str] = None, date_col: Optional[str] = None) -> pd.DataFrame:
    """Load a CSV and return a DataFrame with `day` and `demand` columns.

    Behavior:
    - Attempts to infer a demand column if `demand_col` is not provided by checking
      common names.
    - If `date_col` is provided it will be sorted and `day` becomes a 1-based index
      according to that order. Otherwise the input row order is used.

    Accepts a file path or a file-like object (works with Streamlit uploads).
    """
    df = pd.read_csv(path_or_buffer, parse_dates=[date_col] if date_col else None)

    cols = {c.lower(): c for c in df.columns}

    if demand_col:
        if demand_col not in df.columns and demand_col.lower() in cols:
            demand_col = cols[demand_col.lower()]
        if demand_col not in df.columns:
            raise ValueError(f"Demand column '{demand_col}' not found in CSV")
    else:
        candidates = [
            "demand",
            "demand_qty",
            "demand_quantity",
            "quantity",
            "order_quantity",
            "order_qty",
            "sales",
            "sales_qty",
            "sales_quantity",
            "ordered_quantity",
        ]
        found = None
        for cand in candidates:
            if cand in cols:
                found = cols[cand]
                break
        if not found:
            raise ValueError(
                "Could not infer demand column from CSV. Provide `demand_col` with the column name."
            )
        demand_col = found

    if date_col:
        if date_col not in df.columns and date_col.lower() in cols:
            date_col = cols[date_col.lower()]
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in CSV")
        df = df.sort_values(by=date_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df["day"] = df.index + 1
    df["demand"] = pd.to_numeric(df[demand_col], errors="coerce").fillna(0.0)

    return df[["day", "demand"]]
