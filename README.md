# Demand Forecasting & Inventory Optimization Engine

Python project that forecasts retail demand from historical patterns and feeds predictions into a replenishment policy to reduce stockouts and backlog.

## What it covers
- Synthetic retail demand generation with trend + seasonality + noise
- Forecast model comparison:
  - Naive (last-observation)
  - Moving average
  - Trend + seasonality regression
- MAE-based model selection
- Inventory policy simulation with reorder point and order-up-to logic
- KPI comparison for baseline vs optimized policy

## Technical notes

- Synthetic demand = base level + linear trend + yearly seasonality + weekly seasonality + random noise
- MAE = average absolute difference between actual demand and predicted demand
- Moving average forecast = mean of the last N demand values, updated step by step through the test period
- Trend + seasonality regression = linear regression using day index plus sine/cosine terms for yearly and weekly cycles
- Reorder point = expected demand during lead time + safety stock, computed as `mean * lead_time + z * std * sqrt(lead_time)`
- Order-up-to target = same idea over the full protection period (`lead_time + review_period`)
- Service level = fulfilled demand / total demand including backlog

## Project structure
- `src/data_generation.py` - builds time-series demand data
- `src/forecasting.py` - train/test split, models, MAE evaluation
- `src/inventory.py` - reorder policy and inventory simulation engine
- `src/main.py` - CLI runner and result reporting

## Quick start

1. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Run default experiment:

```bash
python3 -m src.main --days 730 --seed 42
```

3. Run with custom inventory policy settings:

```bash
python3 -m src.main --days 730 --seed 42 --lead-time 12 --service-z 1.95 --review-period 7
```

4. Save outputs for analysis:

```bash
python3 -m src.main --output-eval outputs/forecast_eval.csv --output-inventory outputs/inventory_compare.csv
```

## Streamlit UI

Launch the interactive dashboard:

```bash
streamlit run src/app.py
```

The UI includes:
- Demand generator controls (trend, seasonality, noise)
- Forecast model performance panel with MAE comparison
- Inventory policy controls and KPI comparison vs baseline
- Forecast trace and backlog/inventory trend charts
- CSV downloads for evaluation and inventory comparison
