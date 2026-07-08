"""
RFM (Recency, Frequency, Monetary) customer segmentation via KMeans.

Business use: Tier2/3 customers behave very differently from Tier1 (fewer,
lower-value, more festive-spike-driven orders but huge combined volume).
Segmenting lets marketing target "at-risk high-value" vs "new low-engagement"
customers differently instead of one-size-fits-all push notifications --
directly relevant to Meesho's retention/reactivation campaigns.

Run: python segmentation.py
Outputs: rfm_segments.csv, segment_plot.png
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/customers.csv")

rfm = df[["recency_days", "frequency", "monetary"]].copy()
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# elbow check (printed, not blocking) -- pick k=4 based on typical RFM segment counts
inertias = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(rfm_scaled)
    inertias.append(km.inertia_)
print("Inertia by k (2-7):", [round(i, 1) for i in inertias])

K = 4
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df["segment"] = kmeans.fit_predict(rfm_scaled)

# label segments by their RFM profile (low recency=more recent=better)
profile = df.groupby("segment")[["recency_days", "frequency", "monetary"]].mean()
print("\nSegment profiles:\n", profile)

def label_segment(row):
    if row["recency_days"] < profile["recency_days"].median() and row["monetary"] > profile["monetary"].median():
        return "Champions"
    if row["recency_days"] > profile["recency_days"].median() and row["monetary"] > profile["monetary"].median():
        return "At-Risk High-Value"
    if row["recency_days"] < profile["recency_days"].median() and row["monetary"] <= profile["monetary"].median():
        return "New/Low-Value Active"
    return "Dormant/Churned"

seg_labels = {i: label_segment(profile.loc[i]) for i in profile.index}
df["segment_label"] = df["segment"].map(seg_labels)

df.to_csv("rfm_segments.csv", index=False)

plt.figure(figsize=(7, 5))
for seg in df["segment_label"].unique():
    sub = df[df["segment_label"] == seg]
    plt.scatter(sub["recency_days"], sub["monetary"], s=8, alpha=0.4, label=seg)
plt.xlabel("Recency (days since last order)")
plt.ylabel("Monetary (total spend)")
plt.legend()
plt.title("Customer Segments (RFM + KMeans)")
plt.tight_layout()
plt.savefig("segment_plot.png", dpi=120)

print("\nSegment sizes:\n", df["segment_label"].value_counts())
print("\nSegment sizes by city tier:\n", pd.crosstab(df["city_tier"], df["segment_label"]))
print("\nSaved rfm_segments.csv, segment_plot.png")
