import numpy as np
import pandas as pd

np.random.seed(42)

city_tiers = ["Tier1", "Tier2", "Tier3"]
categories = ["Sarees", "Kurtis", "Western Wear", "Jewellery", "Kids Wear", "Footwear", "Home Furnishing", "Men's Wear"]

# Create synthetic daily sales data for 2 years
start_date = pd.Timestamp("2023-01-01")
end_date = pd.Timestamp("2024-12-31")
dates = pd.date_range(start_date, end_date, freq="D")

rows = []
for city_tier in city_tiers:
    for category in categories:
        # Base trend and seasonality
        base = 120 + (city_tier == "Tier2") * 40 + (city_tier == "Tier3") * 25
        trend = np.linspace(0.9, 1.25, len(dates))
        weekly = 1 + 0.08 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
        seasonal = 1 + 0.1 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
        festive = np.ones(len(dates))
        festive[(dates >= "2024-01-14") & (dates <= "2024-01-20")] = 1.4
        festive[(dates >= "2024-04-09") & (dates <= "2024-04-17")] = 1.35
        festive[(dates >= "2024-07-19") & (dates <= "2024-07-29")] = 1.3
        festive[(dates >= "2024-09-27") & (dates <= "2024-10-27")] = 1.6
        festive[(dates >= "2024-12-21") & (dates <= "2024-12-31")] = 1.45
        noise = np.random.normal(1, 0.08, len(dates))
        orders = np.maximum(0, np.round(base * trend * weekly * seasonal * festive * noise)).astype(int)
        for i, order_count in enumerate(orders):
            rows.append({
                "date": dates[i],
                "city_tier": city_tier,
                "category": category,
                "orders": order_count,
            })

sales_df = pd.DataFrame(rows)
sales_df.to_csv("data/daily_sales.csv", index=False)

# Create synthetic customer data
n_customers = 20000
customers = pd.DataFrame({
    "customer_id": np.arange(1, n_customers + 1),
    "city_tier": np.random.choice(city_tiers, size=n_customers, p=[0.25, 0.45, 0.30]),
})
customers["recency_days"] = np.random.randint(5, 300, size=n_customers)
customers["frequency"] = np.random.poisson(3, size=n_customers) + 1
customers["monetary"] = np.maximum(100, np.random.normal(5000, 1800, size=n_customers).round(2))
customers["monetary"] = np.where(customers["city_tier"] == "Tier1", customers["monetary"] * 1.25, customers["monetary"])
customers["monetary"] = np.where(customers["city_tier"] == "Tier3", customers["monetary"] * 0.8, customers["monetary"])
customers.to_csv("data/customers.csv", index=False)

print("Generated data/daily_sales.csv and data/customers.csv")
