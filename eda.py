"""EDA untuk v2 (multi-order). Jalankan: python eda.py  -> output di reports/"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from common import load_orders

sns.set_style("whitegrid")
OUT = "reports"
os.makedirs(OUT, exist_ok=True)
df = load_orders()

orders_per_cust = df.groupby("Customer_ID").size()
print("=" * 60, "\nRINGKASAN\n" + "=" * 60)
print(f"Shape            : {df.shape}")
print(f"Periode          : {df['Date'].min().date()} s/d {df['Date'].max().date()}")
print(f"Total revenue    : {df['Total_Amount'].sum():,.0f} TRY")
print(f"AOV              : {df['Total_Amount'].mean():,.2f} TRY")
print(f"Customers/Orders : {df['Customer_ID'].nunique():,} / {df['Order_ID'].nunique():,}")
print(f"Order/customer   : mean {orders_per_cust.mean():.2f}, median {orders_per_cust.median():.0f}, max {orders_per_cust.max()}")
print(f"Missing values   : {int(df.isnull().sum().sum())}")

# ---- cek konsistensi flag Is_Returning_Customer vs jumlah order (temuan penting) ----
cust = df.groupby("Customer_ID").agg(n=("Order_ID", "count"), flag_true=("Is_Returning_Customer", "mean"))
print("\nCEK Is_Returning_Customer vs jumlah order asli:")
print(pd.crosstab(cust["n"].clip(upper=3).rename("orders(3=3+)"),
                  df.groupby("Customer_ID")["Is_Returning_Customer"].first().rename("flag_order_pertama")))
print("Customer dengan flag campur (True & False):",
      int(((cust["flag_true"] > 0) & (cust["flag_true"] < 1)).sum()))

num_cols = ["Age", "Unit_Price", "Quantity", "Discount_Amount", "Total_Amount",
            "Session_Duration_Minutes", "Pages_Viewed", "Delivery_Time_Days", "Customer_Rating"]
desc = df[num_cols].describe().T
desc["skew"] = df[num_cols].skew()
desc.to_csv(f"{OUT}/descriptive_stats.csv")
print("\n", desc.round(2))

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
for ax, c in zip(axes.flat, num_cols):
    sns.histplot(df[c], kde=True, ax=ax, color="steelblue"); ax.set_title(c)
plt.tight_layout(); plt.savefig(f"{OUT}/01_distribusi_numerik.png", dpi=110); plt.close()

cat_cols = ["Gender", "City", "Product_Category", "Payment_Method", "Device_Type", "Is_Returning_Customer"]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, c in zip(axes.flat, cat_cols):
    order = df[c].value_counts().index
    sns.countplot(y=df[c], order=order, ax=ax, hue=df[c], hue_order=order, palette="viridis", legend=False)
    ax.set_title(c)
plt.tight_layout(); plt.savefig(f"{OUT}/02_distribusi_kategorik.png", dpi=110); plt.close()

m = df.set_index("Date").resample("ME").agg(revenue=("Total_Amount", "sum"), orders=("Order_ID", "count"))
fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(m.index, m["revenue"], marker="o", color="teal"); ax1.set_ylabel("Revenue (TRY)")
ax2 = ax1.twinx(); ax2.bar(m.index, m["orders"], width=20, alpha=.3, color="orange"); ax2.set_ylabel("Orders")
ax1.set_title("Revenue & Orders Bulanan")
plt.tight_layout(); plt.savefig(f"{OUT}/03_tren_bulanan.png", dpi=110); plt.close()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Korelasi (catatan: Total_Amount = Unit_Price x Qty - Discount)")
plt.tight_layout(); plt.savefig(f"{OUT}/04_korelasi.png", dpi=110); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
df.groupby("Product_Category")["Total_Amount"].sum().sort_values().plot.barh(ax=axes[0], color="coral", title="Revenue / Kategori")
df.groupby("City")["Total_Amount"].sum().sort_values().plot.barh(ax=axes[1], color="skyblue", title="Revenue / Kota")
plt.tight_layout(); plt.savefig(f"{OUT}/05_revenue_kategori_kota.png", dpi=110); plt.close()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.countplot(x=orders_per_cust, ax=axes[0], color="slateblue"); axes[0].set_title("Distribusi order per customer")
sns.boxplot(data=df, x="Device_Type", y="Session_Duration_Minutes", ax=axes[1]); axes[1].set_title("Durasi sesi / device")
sns.boxplot(data=df, x="Customer_Rating", y="Delivery_Time_Days", ax=axes[2]); axes[2].set_title("Delivery vs rating")
plt.tight_layout(); plt.savefig(f"{OUT}/06_perilaku.png", dpi=110); plt.close()

print("\nKEY INSIGHTS")
print(f"- Kategori revenue tertinggi : {df.groupby('Product_Category')['Total_Amount'].sum().idxmax()}")
print(f"- Kota revenue tertinggi     : {df.groupby('City')['Total_Amount'].sum().idxmax()}")
print(f"- Payment terpopuler         : {df['Payment_Method'].value_counts().idxmax()}")
print(f"- Repeat customer (>=2 order): {(orders_per_cust >= 2).mean():.1%}")
print(f"- Share Mobile               : {(df['Device_Type'] == 'Mobile').mean():.1%}")
print(f"- Avg delivery / rating      : {df['Delivery_Time_Days'].mean():.1f} hari / {df['Customer_Rating'].mean():.2f}")
print(f"\nSelesai -> {OUT}/")
