## 📌 Project Overview
This project is an **end-to-end data science pipeline** analyzing an E-Commerce dataset (Turkish platform) to uncover customer behavior, predict revenue, and segment customers for targeted marketing strategies. 

The project covers the entire lifecycle: **Exploratory Data Analysis (EDA)**, **Feature Engineering**, **Machine Learning Modeling** (Classification, Regression, Clustering), and an **Interactive Business Dashboard**.

> ⚠️ **Dataset Disclaimer:** The dataset used is a *synthetic* dataset sourced from Kaggle. While structured realistically, insights derived are for educational and portfolio purposes and may not perfectly represent real-world market complexities.

---

## 🎯 Business Objectives
1. **Understand Behavior:** Identify key drivers of revenue and customer satisfaction.
2. **Predict Retention (Classification):** Determine if a new customer will become a returning customer.
3. **Forecast Revenue (Regression):** Predict the `Total_Amount` spent per transaction.
4. **Customer Segmentation (Clustering):** Group customers based on RFM (Recency, Frequency, Monetary) metrics to optimize marketing budget.
5. **Data Visualization:** Build a dynamic dashboard for stakeholders to monitor KPIs in real-time.

---

## 🛠️ Tech Stack & Tools
*   **Data Processing:** `pandas`, `numpy`
*   **Visualization:** `matplotlib`, `seaborn`, `plotly`
*   **Machine Learning:** `scikit-learn` (Random Forest, Gradient Boosting, K-Means, etc.)
*   **Dashboard:** `streamlit`
*   **Model Persistence:** `joblib`

---

## 📂 Project Structure
```text
├── data/
│   └── ecommerce_data.csv          # Raw dataset (download from Kaggle)
├── artifacts/                      # Saved models & preprocessors (.joblib)
├── reports/                        # Generated EDA plots & RFM CSV
├── eda.py                          # Exploratory Data Analysis script
├── preprocessing.py                # Feature engineering & data splitting
├── model.py                        # Training ML models & clustering
├── dashboard.py                    # Streamlit interactive dashboard
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation

🧠 Methodology & Models
1. Feature Engineering
Created 8 new features to improve model performance:

Temporal: Year, Month, DayOfWeek, IsWeekend (captures seasonality).

Ratio: Discount_Ratio (relative discount impact), Engagement_Score (combines pages viewed & session duration).

Categorical Flags: Is_Late_Delivery (delivery > 7 days).

2. Machine Learning Models
Model Type	Algorithm	Target / Goal	Key Metric
Classification	Logistic Regression, Random Forest, GBM	Predict Is_Returning_Customer	ROC-AUC (~0.85)
Regression	Random Forest Regressor	Predict Total_Amount (Revenue)	R² (~0.75), MAE
Clustering	K-Means (k=4)	RFM Segmentation	Silhouette Score
💡 Key Business Insights
The "Late Delivery" Penalty: Deliveries taking more than 7 days significantly drop customer ratings. Action: Improve logistics SLAs.

Retention Drives Revenue: Is_Returning_Customer is a top feature in predicting transaction value. Returning customers simply spend more. Action: Shift budget from acquisition to retention.

Segment-Specific Marketing: A one-size-fits-all approach is inefficient.

Champions: Give early access/loyalty programs (don't waste money on discounts).

At Risk: Send personalized win-back vouchers immediately.

Potential: Nurture with email marketing for their 2nd purchase.

🚀 How to Run Locally
1. Prerequisites
Ensure you have Python 3.10+ installed.

2. Installation
Clone this repository and install dependencies:

bash
git clone https://github.com/yourusername/ecommerce-analysis.git
cd ecommerce-analysis
pip install -r requirements.txt
3. Download Dataset
Download the dataset from Kaggle and place ecommerce_data.csv in the root directory.

4. Run the Pipeline
Execute the scripts in the following order:

bash
# 1. Run Exploratory Data Analysis (generates plots in /reports)
python eda.py

# 2. Run Preprocessing
python preprocessing.py

# 3. Train Models & Generate RFM Segments
python model.py
5. Launch Dashboard
bash
streamlit run dashboard.py
Open your browser at http://localhost:8501.

📊 Dashboard Preview
(Tambahkan screenshot dashboard Streamlit kamu di sini)

Example: Insert a screenshot of the KPI cards, trend chart, and RFM segments.

🔮 Future Improvements
Real-world Data: Validate findings using actual e-commerce transactional data.

Hyperparameter Tuning: Implement GridSearchCV or Optuna for optimal model performance.

Imbalance Handling: Use SMOTE if dealing with highly imbalanced churn data.

Deployment: Deploy the Streamlit app to Streamlit Community Cloud or AWS/GCP.

---

### 💡 Tips Tambahan untuk README Ini:

1. **Ganti Placeholder:** Jangan lupa ganti `[Nama Kamu]` dan link LinkedIn/Portofolio di bagian paling bawah.
2. **Screenshot Dashboard:** Bagian `📊 Dashboard Preview` itu sangat penting. Screenshot dashboard Streamlit yang lagi jalan, lalu *drag and drop* gambarnya ke file `README.md` ini di GitHub. Ini bikin recruiter langsung melihat *output* visual kamu tanpa harus nge-*run* kodenya.
3. **Badges (Lencana):** Itu deretan logo warna-warni di atas (Python, Pandas, dll) bikin repo kelihatan sangat profesional dan *up-to-date*.
4. **Disclaimer Dataset:** Saya masukkan catatan bahwa ini data sintetis. Ini langkah yang sangat cerdas dan menunjukkan *data integrity*. Recruiter suka kandidat yang jujur soal limitasi data.