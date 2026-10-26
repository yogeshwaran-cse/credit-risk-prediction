# 💳 Credit Score & Risk Prediction Engine

An end-to-end Machine Learning pipeline and interactive underwriting dashboard for assessing retail loan credit risk, estimating default probabilities, and computing credit scores ($300\text{--}850$).

---

## 📌 Overview

In retail banking and consumer lending, evaluating credit risk is essential to minimizing Non-Performing Assets (NPAs) while expanding credit access to creditworthy borrowers.

This project implements an end-to-end credit scoring solution:
1. **Data Science & Modeling Pipeline** (`credit_risk_model.ipynb`): Ingests bureau, customer, and loan data, performs feature engineering, applies regulatory scorecard screening (WoE / IV), handles severe class imbalance via SMOTE-Tomek, tunes models with Optuna and Bayesian Optimization, and evaluates discrimination power (AUC-ROC & Gini).
2. **Prediction Helper** (`prediction_helper.py`): Production inference module that loads serialized artifacts, computes derived financial ratios, applies MinMax scaling, encodes categorical fields, and maps default probability to standard credit scores.
3. **Interactive Web Application** (`main.py`): A clean, minimal Streamlit UI structured in **3 columns across multiple rows** for rapid loan origination and risk underwriting assessment.

---

## 🏗️ Repository Architecture

```plaintext
credit score/
├── artifacts/
│   ├── model.joblib              # Serialized dict: model, scaler, cols_to_scale, features
│   └── scaler.joblib             # Standalone MinMaxScaler artifact
├── dataset/
│   ├── bureau_data.csv           # Bureau history: open accounts, DPD, utilization
│   ├── customers.csv             # Demographics: age, income, residence type
│   └── loans.csv                 # Loan terms: amount, purpose, type, default label
├── credit_risk_model.ipynb       # Documented Jupyter notebook with complete ML workflow
├── prediction_helper.py          # Preprocessing, feature scaling, & scoring engine
├── main.py                       # Streamlit UI (3-column multi-row layout)
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation
```

---

## ⚙️ Machine Learning Pipeline

### 1. Data Ingestion & Relational Merging
* Merges `bureau_data.csv`, `customers.csv`, and `loans.csv` on primary key `cust_id`.
* Employs stratified train-test splitting ($70/30$) to strictly prevent data leakage before transformations.

### 2. Business Anomaly Filtering & Data Hygiene
* **Anomaly Removal**: Filters out loan records where processing fees exceed compliant retail limits.
* **Categorical Normalization**: Corrects categorical inconsistencies (e.g., `'Personaal'` $\rightarrow$ `'Personal'`).
* **Imputation**: Imputes missing entries in `residence_type` using the training distribution mode.

### 3. Domain-Driven Feature Engineering
Derives key financial risk indicators capturing borrower leverage, repayment history, and delinquency severity:

| Feature Name | Formula | Banking Significance |
| :--- | :--- | :--- |
| **Loan-to-Income (LTI)** | $\frac{\text{Loan Amount}}{\text{Annual Income}}$ | Measures borrower leverage & debt absorption capacity |
| **Delinquency Ratio** | $\frac{\text{Delinquent Months}}{\text{Total Loan Months}}$ | Proportion of borrowing history spent in default |
| **Average DPD per Delinquency** | $\frac{\text{Total Days Past Due}}{\text{Delinquent Months}}$ | Severity/depth of payment delays when defaults occur |
| **Credit Utilization per Income** | $\frac{\text{Credit Utilization Ratio}}{\text{LTI}}$ | Credit card usage intensity normalized by leverage |

### 4. Multicollinearity & Feature Screening
* **MinMax Scaling**: Rescales continuous variables to $[0, 1]$ using `MinMaxScaler`.
* **Variance Inflation Factor (VIF)**: Filters out severe multicollinear variables ($VIF > 5$), dropping operational redundancies (`sanction_amount`, `processing_fee`, `gst`, `net_disbursement`, `principal_outstanding`).
* **Information Value (IV) & Weight of Evidence (WoE)**: Implements Basel scorecard standards, retaining features with meaningful predictive power ($IV > 0.02$).

### 5. Class Imbalance Mitigation & Optimization
* Addresses significant class imbalance (~$91.4\%$ Non-Defaults vs ~$8.6\%$ Defaults) using **SMOTE-Tomek** (synthetic over-sampling combined with Tomek links cleaning).
* Hyperparameter tuning across **Logistic Regression** and **XGBoost** utilizing **Optuna** (maximizing Macro F1-score), **Bayesian Optimization**, and **GridSearchCV**.
* Achieved **$94\%$ recall** on default cases and **$\text{Gini} \approx 0.94$** ($\text{AUC} \approx 0.97$).

---

## 📊 Credit Scorecard Calibration

The model estimates the applicant's probability of default $P(\text{Default})$ and calibrates it to a standard credit score range ($300\text{--}850$):

$$\text{Credit Score} = 300 + (1 - P(\text{Default})) \times 550$$

### Underwriting Decision Matrix

| Default Probability | Credit Score Range | Rating Grade | Risk Tier | Automated Underwriting Decision |
| :---: | :---: | :---: | :---: | :--- |
| **$< 15\%$** | $768\text{--}850$ | **Prime (AAA)** | Very Low Risk | **Approved** — Prime interest rate |
| **$15\% - 30\%$** | $685\text{--}767$ | **Good (AA)** | Low Risk | **Approved** — Standard terms |
| **$30\% - 50\%$** | $575\text{--}684$ | **Fair (BBB)** | Moderate Risk | **Conditional Approval** / Manual verification |
| **$50\% - 70\%$** | $465\text{--}574$ | **Subprime (B)** | High Risk | **Review** / Additional collateral required |
| **$> 70\%$** | $300\text{--}464$ | **Poor (D)** | Very High Risk | **Declined** — High default risk |

---


### Test Cases & Scenario Presets
The application includes **5 pre-calibrated benchmark test cases** derived from actual retail portfolio borrower records covering the entire credit risk continuum:
* **🟢 Case 1: Prime (AAA)**: Low utilization (7%), high income, zero delinquencies $\rightarrow$ **Score: 850 | Very Low Risk (0.0% default)**
* **🟢 Case 2: Good (AA)**: Clean repayment track record, home secured loan $\rightarrow$ **Score: 727 | Low Risk (22.3% default)**
* **🟡 Case 3: Fair (BBB)**: Higher leverage, 9 delinquent months, 42 DPD $\rightarrow$ **Score: 646 | Moderate Risk (37.1% default)**
* **🟠 Case 4: Subprime (B)**: Unsecured loan, 96% card usage, 8 delinquent months $\rightarrow$ **Score: 517 | High Risk (60.6% default)**
* **🔴 Case 5: Poor (D)**: Maxed card usage (97%), 18 delinquent months, 85 DPD $\rightarrow$ **Score: 352 | Critical Risk (90.6% default)**

### Interactive UI Features
* **Capsule Pill Selector (`st.pills`) & Sidebar Presets**: 1-click loading of any test case to automatically populate all 12 input fields.
* **Input Layout (3 Columns per Row)**:
  * **Row 1 (Personal & Residence)**: `Age`, `Annual Income (₹)`, `Residence Type`
  * **Row 2 (Loan Terms)**: `Loan Amount (₹)`, `Loan Purpose`, `Loan Type`
  * **Row 3 (Tenure & Credit Usage)**: `Loan Tenure (Months)`, `Credit Utilization Ratio (%)`, `Number of Open Accounts`
  * **Row 4 (Repayment History)**: `Total Loan History (Months)`, `Delinquent Months`, `Total Days Past Due (DPD)`
* **Assessment Output**:
  * **Primary Metric Cards (3 Columns)**: Estimated Credit Score, Default Probability, and Underwriting Recommendation.
  * **Dynamic Credit Score Spectrum Meter**: Visual progress indicator showing borrower's position on the $300\text{--}850$ continuum across credit bands.
  * **Computed Financial Ratios (3 Columns)**: Loan-to-Income (LTI), Delinquency Ratio, Avg DPD per Delinquency.
  * **Benchmark Evaluation Matrix**: Collapsible side-by-side comparison table of all 5 standard test cases.


---

## 🚀 Quick Start Guide

### 1. Prerequisites
* Python 3.10 or higher installed on your machine.

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/credit-score.git
cd "credit score"
```

### 3. Create & Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch the Streamlit Application
```bash
streamlit run main.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 📦 Tech Stack & Dependencies

* **Language**: Python 3.10+
* **Frontend**: [Streamlit](https://streamlit.io/)
* **Data Processing & ML**: `scikit-learn`, `pandas`, `numpy`, `joblib`, `statsmodels`
* **Imbalance Handling**: `imbalanced-learn` (SMOTE-Tomek, RandomUnderSampler)
* **Gradient Boosting**: `xgboost`
* **Hyperparameter Optimization**: `optuna`, `bayesian-optimization`
* **Visualization**: `matplotlib`, `seaborn`

---

## 📄 License
This project is licensed under the MIT License — feel free to use and adapt it for academic or commercial purposes.
