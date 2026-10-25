"""Business Dashboard (v2 multi-order). Jalankan: streamlit run dashboard.py"""
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from common import load_orders, build_rfm

st.set_page_config(page_title="E-Commerce Dashboard", page_icon="📊", layout="wide")


@st.cache_data
def load_data():
    return load_orders()


@st.cache_resource
def load_seg():
    try:
        return joblib.load("artifacts/segmentation.joblib")
    except FileNotFoundError:
        return None


@st.cache_data
def load_scores():
    try:
        return pd.read_csv("reports/customer_scores.csv")
    except FileNotFoundError:
        return None


df = load_data()
seg = load_seg()
scores = load_scores()

# ---------------- SIDEBAR ----------------
st.sidebar.header("🎛️ Filter")
dmin, dmax = df["Date"].min().date(), df["Date"].max().date()
dr = st.sidebar.date_input("Rentang tanggal", (dmin, dmax), min_value=dmin, max_value=dmax)
if len(dr) != 2:
    st.info("Pilih tanggal awal dan akhir.")
    st.stop()
cats = st.sidebar.multiselect("Kategori", sorted(df["Product_Category"].unique()), default=sorted(df["Product_Category"].unique()))
cities = st.sidebar.multiselect("Kota", sorted(df["City"].unique()), default=sorted(df["City"].unique()))
devs = st.sidebar.multiselect("Device", sorted(df["Device_Type"].unique()), default=sorted(df["Device_Type"].unique()))

mask = (df["Date"].dt.date.between(dr[0], dr[1]) & df["Product_Category"].isin(cats)
        & df["City"].isin(cities) & df["Device_Type"].isin(devs))
f = df[mask]
if f.empty:
    st.warning("Tidak ada data untuk filter ini.")
    st.stop()

st.title("📊 E-Commerce Business Dashboard")
st.caption(f"{dr[0]} s/d {dr[1]} • {len(f):,} order • data sintetis (Kaggle, Turki)")

# ---------------- KPI ----------------
rev, n_ord, n_cus = f["Total_Amount"].sum(), f["Order_ID"].nunique(), f["Customer_ID"].nunique()
opc = f.groupby("Customer_ID").size()
repeat_rate = (opc >= 2).mean()
r1 = st.columns(5)
r1[0].metric("💰 Revenue", f"{rev:,.0f} TRY")
r1[1].metric("🛒 Orders", f"{n_ord:,}")
r1[2].metric("👥 Customers", f"{n_cus:,}")
r1[3].metric("💵 AOV", f"{rev / n_ord:,.0f} TRY")
r1[4].metric("🔁 Repeat customer", f"{repeat_rate:.1%}", help="% customer dengan ≥2 order pada filter ini")
r2 = st.columns(5)
r2[0].metric("📦 Order / customer", f"{n_ord / n_cus:.2f}")
r2[1].metric("⭐ Rating", f"{f['Customer_Rating'].mean():.2f}/5")
r2[2].metric("🚚 Delivery", f"{f['Delivery_Time_Days'].mean():.1f} hari")
r2[3].metric("⏰ Late (>7 hari)", f"{f['Is_Late'].mean():.1%}")
r2[4].metric("🎁 Discount rate", f"{f['Discount_Amount'].sum() / f['Gross_Amount'].sum():.1%}", help="Diskon / gross sales")

tab1, tab2, tab3 = st.tabs(["📈 Sales", "👥 Customer & RFM", "🔮 Retention targets"])

# ---------------- TAB 1 ----------------
with tab1:
    c1, c2 = st.columns([2, 1])
    m = f.set_index("Date").resample("ME").agg(Revenue=("Total_Amount", "sum"), Orders=("Order_ID", "count")).reset_index()
    fig = go.Figure()
    fig.add_bar(x=m["Date"], y=m["Orders"], name="Orders", yaxis="y2", opacity=.3, marker_color="orange")
    fig.add_scatter(x=m["Date"], y=m["Revenue"], name="Revenue", mode="lines+markers", line=dict(color="teal", width=3))
    fig.update_layout(title="Revenue & orders bulanan", yaxis=dict(title="TRY"),
                      yaxis2=dict(title="Orders", overlaying="y", side="right"), hovermode="x unified", height=380)
    c1.plotly_chart(fig, use_container_width=True)
    pay = f["Payment_Method"].value_counts().reset_index()
    pay.columns = ["Method", "Orders"]
    c2.plotly_chart(px.pie(pay, values="Orders", names="Method", hole=.5, title="Payment method").update_layout(height=380),
                    use_container_width=True)

    c1, c2 = st.columns(2)
    cat = f.groupby("Product_Category")["Total_Amount"].sum().sort_values().reset_index()
    c1.plotly_chart(px.bar(cat, x="Total_Amount", y="Product_Category", orientation="h", title="Revenue / kategori")
                    .update_layout(height=380), use_container_width=True)
    city = f.groupby("City")["Total_Amount"].sum().sort_values().reset_index()
    c2.plotly_chart(px.bar(city, x="Total_Amount", y="City", orientation="h", title="Revenue / kota")
                    .update_layout(height=380), use_container_width=True)

    c1, c2 = st.columns(2)
    dev = f.groupby("Device_Type").agg(Orders=("Order_ID", "count"), AOV=("Total_Amount", "mean"),
                                       Session=("Session_Duration_Minutes", "mean"), Pages=("Pages_Viewed", "mean")).round(2).reset_index()
    c1.dataframe(dev, use_container_width=True, hide_index=True)
    box = f.assign(Rating=f["Customer_Rating"].astype(str))
    c2.plotly_chart(px.box(box, x="Rating", y="Delivery_Time_Days", title="Delivery time vs rating",
                           category_orders={"Rating": ["1", "2", "3", "4", "5"]}).update_layout(height=320),
                    use_container_width=True)

    summ = f.groupby("Product_Category").agg(Orders=("Order_ID", "count"), Revenue=("Total_Amount", "sum"),
                                             AOV=("Total_Amount", "mean"), Rating=("Customer_Rating", "mean"),
                                             Delivery=("Delivery_Time_Days", "mean")).round(2).sort_values("Revenue", ascending=False)
    st.subheader("🏆 Ringkasan kategori")
    st.dataframe(summ, use_container_width=True)

# ---------------- TAB 2 ----------------
with tab2:
    if seg is None:
        st.info("Jalankan `python model.py` dulu untuk membuat model segmentasi.")
    else:
        # RFM dihitung ulang dari data terfilter, pakai snapshot + scaler + kmeans dari training
        rfm = build_rfm(f, seg["snapshot"])
        Z = rfm[["Recency", "Frequency", "Monetary"]].copy()
        Z["Frequency"], Z["Monetary"] = np.log1p(Z["Frequency"]), np.log1p(Z["Monetary"])
        rfm["Segment"] = pd.Series(seg["kmeans"].predict(seg["scaler"].transform(Z))).map(seg["label_map"]).values
        st.caption("Segmen dihitung dari order pada filter aktif (model K-Means dari full data). "
                   "Kalau filter dipersempit, nilai Frequency/Monetary mengecil sehingga customer bisa pindah segmen.")
        agg = rfm.groupby("Segment").agg(Customers=("Customer_ID", "count"), Recency=("Recency", "mean"),
                                         Frequency=("Frequency", "mean"), Monetary=("Monetary", "mean"),
                                         Revenue=("Monetary", "sum")).round(1).reset_index()
        agg["Rev_%"] = (agg["Revenue"] / agg["Revenue"].sum() * 100).round(1)
        c1, c2 = st.columns([1, 2])
        c1.plotly_chart(px.pie(agg, values="Customers", names="Segment", hole=.4).update_layout(height=340),
                        use_container_width=True)
        c2.dataframe(agg.sort_values("Revenue", ascending=False), use_container_width=True, hide_index=True, height=300)
        st.markdown("""
| Segmen | Aksi |
|---|---|
| **Champions** | Loyalty program, early access, jangan over-diskon |
| **Loyal** | Bundling / cross-sell kategori lain |
| **Potential** | Nurture: rekomendasi produk + email follow-up order ke-2/3 |
| **At Risk** | Win-back campaign (diskon personal, reminder) |
""")

# ---------------- TAB 3 ----------------
with tab3:
    if scores is None:
        st.info("Jalankan `python model.py` dulu untuk membuat customer_scores.csv.")
    else:
        st.warning("Model repurchase di data ini lemah (AUC ≈ 0.59) — pakai sebagai ranking kasar, bukan prediksi akurat. "
                   "Data sintetis: perilaku beli ulang hampir acak.")
        seg_pick = st.multiselect("Segmen", sorted(scores["Segment"].dropna().unique()),
                                  default=sorted(scores["Segment"].dropna().unique()))
        topn = st.slider("Tampilkan N customer", 10, 200, 50, step=10)
        sc = scores[scores["Segment"].isin(seg_pick)].sort_values(["Monetary"], ascending=False)
        sc = sc[sc["Repurchase_Prob"] < sc["Repurchase_Prob"].median()].head(topn)
        st.caption("Customer bernilai tinggi (Monetary) dengan probabilitas beli ulang di bawah median → prioritas retention.")
        st.dataframe(sc.round(3), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download daftar (CSV)", sc.to_csv(index=False).encode(), "retention_targets.csv", "text/csv")

st.divider()
st.download_button("⬇️ Download order terfilter (CSV)", f.to_csv(index=False).encode(), "orders_filtered.csv", "text/csv")
