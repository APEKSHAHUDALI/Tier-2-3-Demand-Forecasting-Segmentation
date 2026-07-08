"""
Forecasts daily order demand per (city_tier, category) using Prophet, with
Indian festive-season windows added as custom holiday regressors.

Business use: demand forecasting drives inventory pre-positioning and seller
onboarding pushes ahead of festive spikes -- getting this wrong means either
stockouts during Diwali (lost GMV) or excess unsold inventory after (working
capital locked up in Tier 2/3 warehouses).

Run: python forecast.py --city_tier Tier2 --category Sarees --horizon 60
"""
import argparse
import pandas as pd
from prophet import Prophet
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FESTIVE_WINDOWS_2024_2026 = [
    ("2024-01-14", "2024-01-20"), ("2024-04-09", "2024-04-17"),
    ("2024-07-19", "2024-07-29"), ("2024-09-27", "2024-10-27"), ("2024-12-21", "2024-12-31"),
    ("2025-01-14", "2025-01-20"), ("2025-04-09", "2025-04-17"),
    ("2025-07-19", "2025-07-29"), ("2025-09-27", "2025-10-27"), ("2025-12-21", "2025-12-31"),
]

def build_holidays():
    rows = []
    for start, end in FESTIVE_WINDOWS_2024_2026:
        rows.append({"holiday": "festive_sale_window", "ds": start,
                     "lower_window": 0, "upper_window": (pd.Timestamp(end) - pd.Timestamp(start)).days})
    return pd.DataFrame(rows)


def run_forecast(city_tier, category, horizon_days=60, plot=True):
    df = pd.read_csv("data/daily_sales.csv")
    sub = df[(df["city_tier"] == city_tier) & (df["category"] == category)][["date", "orders"]]
    sub = sub.rename(columns={"date": "ds", "orders": "y"})
    sub["ds"] = pd.to_datetime(sub["ds"])

    model = Prophet(holidays=build_holidays(), yearly_seasonality=True,
                     weekly_seasonality=True, changepoint_prior_scale=0.2)
    model.fit(sub)

    future = model.make_future_dataframe(periods=horizon_days)
    forecast = model.predict(future)

    if plot:
        fig = model.plot(forecast)
        plt.title(f"Demand forecast: {category} — {city_tier}")
        plt.xlabel("Date"); plt.ylabel("Orders/day")
        plt.tight_layout()
        fname = f"forecast_{city_tier}_{category.replace(' ', '_')}.png"
        fig.savefig(fname, dpi=120)
        print(f"Saved {fname}")

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon_days)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city_tier", default="Tier2")
    parser.add_argument("--category", default="Sarees")
    parser.add_argument("--horizon", type=int, default=60)
    args = parser.parse_args()

    result = run_forecast(args.city_tier, args.category, args.horizon)
    print(result.tail(10).to_string(index=False))
