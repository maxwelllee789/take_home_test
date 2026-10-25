"""Customer-level feature table + split. Jalankan: python preprocessing.py
Unit analisis = 1 customer. Fitur dari order < CUTOFF, target dari order >= CUTOFF.
"""
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from common import load_orders, customer_features, CUTOFF

os.makedirs("artifacts", exist_ok=True)
os.makedirs("reports", exist_ok=True)

df = load_orders()
cust = customer_features(df)
cust.to_csv("reports/customer_features.csv", index=False)

num_features = ["Recency", "Tenure_Days", "Frequency", "Monetary", "AOV", "Avg_Quantity",
                "Avg_Rating", "Avg_Delivery", "Late_Share", "Avg_Session", "Avg_Pages",
                "N_Categories", "Mobile_Share", "Discount_Ratio", "Age"]
cat_features = ["Gender", "City"]

X = cust[num_features + cat_features]
y_clf = cust["Will_Repurchase"]
y_reg = cust["Future_Spend"]

idx_train, idx_test = train_test_split(cust.index, test_size=0.2, random_state=42, stratify=y_clf)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_features),
])
preprocessor.fit(X.loc[idx_train])

print(f"Cutoff           : {CUTOFF.date()}")
print(f"Customers        : {len(cust):,} (train {len(idx_train):,} / test {len(idx_test):,})")
print(f"Repurchase rate  : {y_clf.mean():.1%}")
print(f"Future spend     : mean {y_reg.mean():,.0f}, median {y_reg.median():,.0f} TRY")

joblib.dump(
    {"preprocessor": preprocessor, "num_features": num_features, "cat_features": cat_features,
     "idx_train": idx_train, "idx_test": idx_test},
    "artifacts/preprocessor.joblib",
)
print("OK -> artifacts/preprocessor.joblib, reports/customer_features.csv")
