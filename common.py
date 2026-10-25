"""Shared config + helper (dipakai eda.py, preprocessing.py, model.py, dashboard.py)."""
import pandas as pd
import numpy as np

DATA_PATH = "data/ecommerce_customer_behavior_dataset_v2.csv"

# Cutoff waktu untuk model prediktif: fitur dari order SEBELUM cutoff,
# target dari order pada/sesudah cutoff -> tidak ada data leakage.
CUTOFF = pd.Timestamp("2023-12-01")

LATE_DAYS = 7


def load_orders(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Is_Returning_Customer"] = df["Is_Returning_Customer"].astype(str).str.lower().eq("true")
    df["Gross_Amount"] = df["Unit_Price"] * df["Quantity"]
    df["Is_Late"] = (df["Delivery_Time_Days"] > LATE_DAYS).astype(int)
    return df.sort_values(["Customer_ID", "Date"]).reset_index(drop=True)


def build_rfm(df: pd.DataFrame, snapshot: pd.Timestamp | None = None) -> pd.DataFrame:
    snapshot = snapshot or (df["Date"].max() + pd.Timedelta(days=1))
    return df.groupby("Customer_ID").agg(
        Recency=("Date", lambda s: (snapshot - s.max()).days),
        Frequency=("Order_ID", "nunique"),
        Monetary=("Total_Amount", "sum"),
    ).reset_index()


def customer_features(df: pd.DataFrame, cutoff: pd.Timestamp = CUTOFF) -> pd.DataFrame:
    """Satu baris per customer, hanya dari order sebelum cutoff."""
    pre = df[df["Date"] < cutoff]
    g = pre.groupby("Customer_ID")
    feat = g.agg(
        Recency=("Date", lambda s: (cutoff - s.max()).days),
        Tenure_Days=("Date", lambda s: (cutoff - s.min()).days),
        Frequency=("Order_ID", "nunique"),
        Monetary=("Total_Amount", "sum"),
        AOV=("Total_Amount", "mean"),
        Avg_Quantity=("Quantity", "mean"),
        Avg_Rating=("Customer_Rating", "mean"),
        Avg_Delivery=("Delivery_Time_Days", "mean"),
        Late_Share=("Is_Late", "mean"),
        Avg_Session=("Session_Duration_Minutes", "mean"),
        Avg_Pages=("Pages_Viewed", "mean"),
        N_Categories=("Product_Category", "nunique"),
        Discount_Total=("Discount_Amount", "sum"),
        Gross_Total=("Gross_Amount", "sum"),
        Mobile_Share=("Device_Type", lambda s: (s == "Mobile").mean()),
        Age=("Age", "first"),
        Gender=("Gender", "first"),
        City=("City", "first"),
    )
    feat["Discount_Ratio"] = feat["Discount_Total"] / feat["Gross_Total"]
    feat = feat.drop(columns=["Discount_Total", "Gross_Total"]).reset_index()

    post = df[df["Date"] >= cutoff].groupby("Customer_ID").agg(
        Future_Orders=("Order_ID", "nunique"),
        Future_Spend=("Total_Amount", "sum"),
    ).reset_index()
    feat = feat.merge(post, on="Customer_ID", how="left").fillna({"Future_Orders": 0, "Future_Spend": 0})
    feat["Will_Repurchase"] = (feat["Future_Orders"] > 0).astype(int)
    return feat
