import streamlit as st
import pandas as pd
import prediction_helper as ph

# Page configuration
st.set_page_config(
    page_title="Credit Score & Risk Predictor",
    page_icon="💳",
    layout="wide"
)

# Minimal custom styling
st.markdown("""
<style>
    .main-title {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .sub-title {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 24px;
    }
    .metric-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    @media (prefers-color-scheme: dark) {
        .metric-box {
            background-color: #1e293b;
            border-color: #334155;
        }
    }
    .metric-title {
        font-size: 13px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .metric-score {
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .metric-sub {
        font-size: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">💳 Credit Score & Risk Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter applicant and loan details across the form below to evaluate default risk and calculate credit score.</div>', unsafe_allow_html=True)

# ----------------- 3 Columns, Multiple Rows -----------------

# Row 1: Personal & Residence Details
row1_col1, row1_col2, row1_col3 = st.columns(3)
with row1_col1:
    age = st.number_input("Age", min_value=18, max_value=90, value=35, step=1)

with row1_col2:
    income = st.number_input("Annual Income (₹)", min_value=10000, max_value=50000000, value=800000, step=25000)

with row1_col3:
    residence_type = st.selectbox("Residence Type", options=["Owned", "Rented", "Mortgage"], index=0)

# Row 2: Loan Amount, Purpose & Type
row2_col1, row2_col2, row2_col3 = st.columns(3)
with row2_col1:
    loan_amount = st.number_input("Loan Amount (₹)", min_value=10000, max_value=25000000, value=300000, step=10000)

with row2_col2:
    loan_purpose = st.selectbox("Loan Purpose", options=["Personal", "Home", "Education", "Auto"], index=0)

with row2_col3:
    loan_type = st.selectbox("Loan Type", options=["Secured", "Unsecured"], index=0)

# Row 3: Loan Tenure, Credit Utilization & Open Accounts
row3_col1, row3_col2, row3_col3 = st.columns(3)
with row3_col1:
    loan_tenure_months = st.number_input("Loan Tenure (Months)", min_value=6, max_value=120, value=36, step=6)

with row3_col2:
    credit_utilization_ratio = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=25.0, step=1.0)

with row3_col3:
    number_of_open_accounts = st.number_input("Number of Open Accounts", min_value=1, max_value=4, value=2, step=1)

# Row 4: Past Loan History & Delinquencies
row4_col1, row4_col2, row4_col3 = st.columns(3)
with row4_col1:
    total_loan_months = st.number_input("Total Loan History (Months)", min_value=1, max_value=120, value=36, step=6)

with row4_col2:
    delinquent_months = st.number_input("Delinquent Months", min_value=0, max_value=60, value=0, step=1)

with row4_col3:
    total_dpd = st.number_input("Total Days Past Due (DPD)", min_value=0, max_value=365, value=0, step=5)

st.write("")

# ----------------- Predict Button -----------------
if st.button("Calculate Credit Score", type="primary"):
    try:
        result = ph.predict(
            age=age,
            income=income,
            loan_amount=loan_amount,
            loan_tenure_months=loan_tenure_months,
            credit_utilization_ratio=credit_utilization_ratio,
            number_of_open_accounts=number_of_open_accounts,
            total_loan_months=total_loan_months,
            delinquent_months=delinquent_months,
            total_dpd=total_dpd,
            residence_type=residence_type,
            loan_purpose=loan_purpose,
            loan_type=loan_type
        )

        st.divider()

        # Primary Results in 3 Columns
        res1, res2, res3 = st.columns(3)

        with res1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">Estimated Credit Score</div>
                <div class="metric-score" style="color: {result['status_color']};">{result['credit_score']}</div>
                <div class="metric-sub" style="color: {result['status_color']};">{result['rating']}</div>
            </div>
            """, unsafe_allow_html=True)

        with res2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">Default Probability</div>
                <div class="metric-score" style="color: {result['status_color']};">{result['probability_pct']}%</div>
                <div class="metric-sub" style="color: {result['status_color']};">{result['risk_tier']}</div>
            </div>
            """, unsafe_allow_html=True)

        with res3:
            decision_text = "Approved" if result['prediction'] == 0 else "High Risk"
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">Recommendation</div>
                <div class="metric-score" style="color: {result['status_color']}; font-size: 22px; padding-top: 4px;">{decision_text}</div>
                <div class="metric-sub" style="color: {result['status_color']};">{result['recommendation']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # Financial Ratios in 3 Columns
        ratio1, ratio2, ratio3 = st.columns(3)
        with ratio1:
            st.metric("Loan-to-Income (LTI)", f"{result['loan_to_income']}x")
        with ratio2:
            st.metric("Delinquency Ratio", f"{result['delinquency_ratio'] * 100:.1f}%")
        with ratio3:
            st.metric("Avg DPD per Delinquency", f"{result['avg_dpd_per_delinquency']} days")

    except Exception as e:
        st.error(f"Error computing prediction: {e}")
