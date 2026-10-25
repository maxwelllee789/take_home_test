"""Training 3 model. Jalankan: python model.py
1. Klasifikasi : customer akan beli lagi setelah CUTOFF? (Will_Repurchase)
2. Regresi     : berapa spend customer setelah CUTOFF? (Future_Spend)
3. Clustering  : segmentasi RFM (K-Means) di seluruh data
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (classification_report, confusion_matrix, mean_absolute_error,
                             r2_score, roc_auc_score, silhouette_score)
from sklearn.preprocessing import StandardScaler
from common import load_orders, build_rfm

os.makedirs("artifacts", exist_ok=True)
os.makedirs("reports", exist_ok=True)

df = load_orders()
cust = pd.read_csv("reports/customer_features.csv")
pp = joblib.load("artifacts/preprocessor.joblib")
prep, num_f, cat_f = pp["preprocessor"], pp["num_features"], pp["cat_features"]
tr, te = pp["idx_train"], pp["idx_test"]
X = cust[num_f + cat_f]
Xtr, Xte = prep.transform(X.loc[tr]), prep.transform(X.loc[te])
feat_names = num_f + list(prep.named_transformers_["cat"].get_feature_names_out(cat_f))

# ================= 1. KLASIFIKASI =================
print("=" * 60, "\nMODEL 1: PREDIKSI REPURCHASE\n" + "=" * 60)
ytr, yte = cust.loc[tr, "Will_Repurchase"], cust.loc[te, "Will_Repurchase"]
clfs = {
    "Baseline(majority)": DummyClassifier(strategy="prior"),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42),
}
rows, fitted = [], {}
for name, m in clfs.items():
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    rows.append({"model": name, "AUC": roc_auc_score(yte, p), "Accuracy": (m.predict(Xte) == yte).mean()})
    fitted[name] = m
res = pd.DataFrame(rows).sort_values("AUC", ascending=False)
print(res.round(4).to_string(index=False))
res.to_csv("reports/classifier_comparison.csv", index=False)

best_name = res[res["model"] != "Baseline(majority)"].iloc[0]["model"]
best = fitted[best_name]
print(f"\nBest: {best_name}\n", classification_report(yte, best.predict(Xte)))
joblib.dump(best, "artifacts/classifier.joblib")

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(yte, best.predict(Xte)), annot=True, fmt="d", cmap="Blues",
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.title(f"Confusion Matrix - {best_name}"); plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout(); plt.savefig("reports/08_confusion_matrix.png", dpi=110); plt.close()

# ================= 2. REGRESI =================
print("\n" + "=" * 60, "\nMODEL 2: PREDIKSI FUTURE SPEND (log1p)\n" + "=" * 60)
rtr, rte = np.log1p(cust.loc[tr, "Future_Spend"]), cust.loc[te, "Future_Spend"]
regs = {
    "Baseline(mean)": DummyRegressor(strategy="mean"),
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=10, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42),
}
rrows, rfit = [], {}
for name, m in regs.items():
    m.fit(Xtr, rtr)
    pred = np.expm1(m.predict(Xte))
    rrows.append({"model": name, "MAE_TRY": mean_absolute_error(rte, pred),
                  "R2_TRY": r2_score(rte, pred),
                  "R2_log": r2_score(np.log1p(rte), m.predict(Xte))})
    rfit[name] = m
rres = pd.DataFrame(rrows).sort_values("MAE_TRY")
print(rres.round(3).to_string(index=False))
rres.to_csv("reports/regressor_comparison.csv", index=False)
rbest_name = rres[~rres["model"].str.startswith("Baseline")].sort_values("R2_log", ascending=False).iloc[0]["model"]
reg = rfit[rbest_name]
print(f"Best: {rbest_name}")
joblib.dump(reg, "artifacts/regressor.joblib")

if hasattr(reg, "feature_importances_"):
    imp = pd.DataFrame({"feature": feat_names, "importance": reg.feature_importances_}) \
            .sort_values("importance", ascending=False).head(15)
    plt.figure(figsize=(9, 6))
    sns.barplot(data=imp, x="importance", y="feature", hue="feature", palette="viridis", legend=False)
    plt.title(f"Top Feature Importance - {rbest_name}")
    plt.tight_layout(); plt.savefig("reports/09_feature_importance.png", dpi=110); plt.close()

# ================= 3. CLUSTERING RFM =================
print("\n" + "=" * 60, "\nMODEL 3: SEGMENTASI RFM (K-Means)\n" + "=" * 60)
snapshot = df["Date"].max() + pd.Timedelta(days=1)
rfm = build_rfm(df, snapshot)
Z = rfm[["Recency", "Frequency", "Monetary"]].copy()
Z["Monetary"], Z["Frequency"] = np.log1p(Z["Monetary"]), np.log1p(Z["Frequency"])  # skew tinggi
scaler = StandardScaler().fit(Z)
Zs = scaler.transform(Z)

ks, inertia, sils = range(2, 9), [], []
for k in ks:
    km_k = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Zs)
    inertia.append(km_k.inertia_); sils.append(silhouette_score(Zs, km_k.labels_))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(list(ks), inertia, marker="o"); ax[0].set_title("Elbow (inertia)")
ax[1].plot(list(ks), sils, marker="o", color="green"); ax[1].set_title("Silhouette")
plt.tight_layout(); plt.savefig("reports/10_elbow_silhouette.png", dpi=110); plt.close()
print("Silhouette per k:", {k: round(s, 3) for k, s in zip(ks, sils)})

K = 4  # dipilih agar segmen actionable; cek grafik silhouette
km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(Zs)
rfm["Cluster"] = km.labels_
prof = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()

# Label berdasar peringkat centroid (bukan 1 baris sampel seperti versi sebelumnya)
score = prof["Recency"].rank(ascending=False) + prof["Frequency"].rank() + prof["Monetary"].rank()
order = score.sort_values(ascending=False).index.tolist()
label_map = {order[0]: "Champions", order[-1]: "At Risk"}
mid = order[1:-1]
if len(mid) == 2:
    hi_f = prof.loc[mid, "Frequency"].idxmax()
    label_map[hi_f] = "Loyal"
    label_map[[c for c in mid if c != hi_f][0]] = "Potential"
else:
    for c in mid: label_map[c] = "Potential"
rfm["Segment"] = rfm["Cluster"].map(label_map)

summary = rfm.groupby("Segment").agg(Customers=("Customer_ID", "count"), Recency=("Recency", "mean"),
                                     Frequency=("Frequency", "mean"), Monetary=("Monetary", "mean"),
                                     TotalRevenue=("Monetary", "sum")).round(1)
summary["Rev_Share_%"] = (summary["TotalRevenue"] / summary["TotalRevenue"].sum() * 100).round(1)
print(f"\nSilhouette (k={K}): {silhouette_score(Zs, km.labels_):.3f}\n")
print(summary.sort_values("TotalRevenue", ascending=False).to_string())
rfm.to_csv("reports/rfm_segments.csv", index=False)
summary.to_csv("reports/rfm_summary.csv")
joblib.dump({"kmeans": km, "scaler": scaler, "label_map": label_map, "snapshot": snapshot},
            "artifacts/segmentation.joblib")

# ===== skor customer (untuk dashboard: daftar target retention) =====
cust["Repurchase_Prob"] = best.predict_proba(prep.transform(X))[:, 1]
cust["Split"] = np.where(cust.index.isin(te), "test", "train")
cust.merge(rfm[["Customer_ID", "Segment"]], on="Customer_ID", how="left") \
    [["Customer_ID", "Recency", "Frequency", "Monetary", "Repurchase_Prob", "Will_Repurchase", "Split", "Segment"]] \
    .to_csv("reports/customer_scores.csv", index=False)
print("\nOK -> artifacts/, reports/")
