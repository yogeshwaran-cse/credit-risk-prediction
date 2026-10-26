import streamlit as st
import pandas as pd
import prediction_helper as ph

# Page configuration
st.set_page_config(
    page_title="Credit Score & Risk Predictor",
    page_icon="💳",
    layout="wide"
)

# ----------------- Curated Benchmark Test Cases -----------------
TEST_CASES = {
    "🟢 Case 1: Prime (AAA)": {
        "full_name": "Test Case 1: Prime (AAA) - Ideal Borrower",
        "short_name": "Prime (AAA)",
        "badge_color": "#10b981",
        "expected_score": 850,
        "expected_prob": "0.0%",
        "expected_rating": "Prime (AAA)",
        "expected_tier": "Very Low Risk",
        "expected_decision": "Approved - Prime Terms",
        "description": "High income (₹25.86L), low utilization (7%), zero delinquencies, secured auto loan.",
        "data": {
            "age": 44,
            "income": 2586000,
            "residence_type": "Owned",
            "loan_amount": 2467000,
            "loan_purpose": "Auto",
            "loan_type": "Secured",
            "loan_tenure_months": 33,
            "credit_utilization_ratio": 7.0,
            "number_of_open_accounts": 1,
            "total_loan_months": 42,
            "delinquent_months": 0,
            "total_dpd": 0
        }
    },
    "🟢 Case 2: Good (AA)": {
        "full_name": "Test Case 2: Good (AA) - Low Risk",
        "short_name": "Good (AA)",
        "badge_color": "#34d399",
        "expected_score": 727,
        "expected_prob": "22.3%",
        "expected_rating": "Good (AA)",
        "expected_tier": "Low Risk",
        "expected_decision": "Approved - Standard Rate",
        "description": "Established homeowner, high income (₹36.12L), clean repayment track record, secured home loan.",
        "data": {
            "age": 49,
            "income": 3612000,
            "residence_type": "Owned",
            "loan_amount": 10122000,
            "loan_purpose": "Home",
            "loan_type": "Secured",
            "loan_tenure_months": 44,
            "credit_utilization_ratio": 86.0,
            "number_of_open_accounts": 1,
            "total_loan_months": 10,
            "delinquent_months": 0,
            "total_dpd": 0
        }
    },
    "🟡 Case 3: Fair (BBB)": {
        "full_name": "Test Case 3: Fair (BBB) - Moderate Risk",
        "short_name": "Fair (BBB)",
        "badge_color": "#f59e0b",
        "expected_score": 646,
        "expected_prob": "37.1%",
        "expected_rating": "Fair (BBB)",
        "expected_tier": "Moderate Risk",
        "expected_decision": "Conditional Approval / Manual Review",
        "description": "Tenant borrower with moderate income, 9 delinquent months, and 42 days past due history.",
        "data": {
            "age": 35,
            "income": 1657000,
            "residence_type": "Rented",
            "loan_amount": 5618000,
            "loan_purpose": "Home",
            "loan_type": "Secured",
            "loan_tenure_months": 27,
            "credit_utilization_ratio": 28.0,
            "number_of_open_accounts": 1,
            "total_loan_months": 41,
            "delinquent_months": 9,
            "total_dpd": 42
        }
    },
    "🟠 Case 4: Subprime (B)": {
        "full_name": "Test Case 4: Subprime (B) - High Risk",
        "short_name": "Subprime (B)",
        "badge_color": "#f97316",
        "expected_score": 517,
        "expected_prob": "60.6%",
        "expected_rating": "Subprime (B)",
        "expected_tier": "High Risk",
        "expected_decision": "Review / Additional Collateral Required",
        "description": "Unsecured personal loan applicant with near-maxed card usage (96%) and 8 delinquent months.",
        "data": {
            "age": 55,
            "income": 3147000,
            "residence_type": "Owned",
            "loan_amount": 1700000,
            "loan_purpose": "Personal",
            "loan_type": "Unsecured",
            "loan_tenure_months": 12,
            "credit_utilization_ratio": 96.0,
            "number_of_open_accounts": 2,
            "total_loan_months": 36,
            "delinquent_months": 8,
            "total_dpd": 33
        }
    },
    "🔴 Case 5: Poor (D)": {
        "full_name": "Test Case 5: Poor (D) - Default Risk",
        "short_name": "Poor (D)",
        "badge_color": "#ef4444",
        "expected_score": 352,
        "expected_prob": "90.6%",
        "expected_rating": "Poor (D)",
        "expected_tier": "Very High Risk",
        "expected_decision": "Decline - High Probability of Default",
        "description": "Young borrower (age 20) with 97% card utilization, 18 delinquent months, 85 DPD, unsecured loan.",
        "data": {
            "age": 20,
            "income": 3748000,
            "residence_type": "Owned",
            "loan_amount": 2969000,
            "loan_purpose": "Personal",
            "loan_type": "Unsecured",
            "loan_tenure_months": 14,
            "credit_utilization_ratio": 97.0,
            "number_of_open_accounts": 2,
            "total_loan_months": 63,
            "delinquent_months": 18,
            "total_dpd": 85
        }
    }
}

DEFAULT_VALUES = {
    "age": 35,
    "income": 800000,
    "residence_type": "Owned",
    "loan_amount": 300000,
    "loan_purpose": "Personal",
    "loan_type": "Secured",
    "loan_tenure_months": 36,
    "credit_utilization_ratio": 25.0,
    "number_of_open_accounts": 2,
    "total_loan_months": 36,
    "delinquent_months": 0,
    "total_dpd": 0
}

# ----------------- State Management -----------------
for key, val in DEFAULT_VALUES.items():
    if f"field_{key}" not in st.session_state:
        st.session_state[f"field_{key}"] = val

if "active_case_key" not in st.session_state:
    st.session_state["active_case_key"] = None

if "trigger_prediction" not in st.session_state:
    st.session_state["trigger_prediction"] = False


def load_test_case_data(case_key: str):
    """Callback to update all input fields in session state from selected test case."""
    if case_key in TEST_CASES:
        for k, v in TEST_CASES[case_key]["data"].items():
            st.session_state[f"field_{k}"] = v
        st.session_state["active_case_key"] = case_key
        st.session_state["trigger_prediction"] = True
    elif case_key in ["reset", "↺ Custom / Reset"]:
        for k, v in DEFAULT_VALUES.items():
            st.session_state[f"field_{k}"] = v
        st.session_state["active_case_key"] = None
        st.session_state["trigger_prediction"] = False


def on_pill_changed():
    val = st.session_state.get("pills_selection")
    if val:
        load_test_case_data(val)


# ----------------- Minimal Custom Styling -----------------
st.markdown("""
<style>
    .main-title {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .sub-title {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 16px;
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
    .gauge-wrapper {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 14px;
        margin-bottom: 14px;
    }
    @media (prefers-color-scheme: dark) {
        .gauge-wrapper {
            background: #0f172a;
            border-color: #334155;
        }
    }
    .gauge-bar-bg {
        height: 12px;
        background: linear-gradient(to right, #ef4444 0%, #f97316 25%, #f59e0b 50%, #34d399 75%, #10b981 100%);
        border-radius: 6px;
        position: relative;
        margin-top: 8px;
        margin-bottom: 8px;
    }
    .gauge-pointer {
        position: absolute;
        top: -6px;
        width: 6px;
        height: 24px;
        background-color: #0f172a;
        border: 2px solid #ffffff;
        border-radius: 3px;
        box-shadow: 0 0 6px rgba(0,0,0,0.45);
        transform: translateX(-50%);
    }
    @media (prefers-color-scheme: dark) {
        .gauge-pointer {
            background-color: #ffffff;
            border-color: #0f172a;
        }
    }
    .gauge-labels {
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        color: #64748b;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Sidebar Quick Controls -----------------
with st.sidebar:
    st.markdown("### 🧪 Test Cases Preset")
    st.caption("Select a pre-calibrated scenario from the credit risk dataset to immediately populate and evaluate.")
    
    dropdown_options = ["-- Select a Test Case --"] + list(TEST_CASES.keys())
    selected_from_dropdown = st.selectbox(
        "Choose Test Scenario",
        options=dropdown_options,
        index=0,
        key="dropdown_test_case"
    )

    col_load, col_rst = st.columns(2)
    with col_load:
        if st.button("📥 Load", use_container_width=True, type="primary"):
            if selected_from_dropdown in TEST_CASES:
                load_test_case_data(selected_from_dropdown)
                st.rerun()
    with col_rst:
        if st.button("↺ Reset", use_container_width=True):
            load_test_case_data("reset")
            st.rerun()

    st.divider()
    st.markdown("#### 🎯 Scorecard Grading")
    st.markdown("""
    - **768 - 850**: 🟢 Prime (AAA) — Prime Rate
    - **685 - 767**: 🟢 Good (AA) — Standard Terms
    - **575 - 684**: 🟡 Fair (BBB) — Review
    - **465 - 574**: 🟠 Subprime (B) — Collateral
    - **300 - 464**: 🔴 Poor (D) — Decline
    """)

# ----------------- Main Header -----------------
st.markdown('<div class="main-title">💳 Credit Score & Risk Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter applicant and loan details across the 3-column form below, or load benchmark test cases to evaluate default probability.</div>', unsafe_allow_html=True)

# ----------------- Interactive Test Cases Pill Selector -----------------
pill_options = list(TEST_CASES.keys()) + ["↺ Custom / Reset"]
current_pill_default = st.session_state["active_case_key"] if st.session_state["active_case_key"] in TEST_CASES else "↺ Custom / Reset"

selected_pill = st.pills(
    "⚡ Quick Benchmark Test Cases",
    options=pill_options,
    default=current_pill_default,
    key="pills_selection",
    on_change=on_pill_changed
)

# Active Profile Info Banner
active_key = st.session_state["active_case_key"]
if active_key in TEST_CASES:
    info = TEST_CASES[active_key]
    st.info(f"**Loaded {info['full_name']}** — {info['description']} *(Expected Score: {info['expected_score']} | Rating: {info['expected_rating']})*")
else:
    st.caption("ℹ️ Form active with custom values. You can tweak any input below or select a benchmark test case above.")

st.write("")

# ----------------- 3 Columns, Multiple Rows (Preserved Exact Layout) -----------------

# Row 1: Personal & Residence Details
row1_col1, row1_col2, row1_col3 = st.columns(3)
with row1_col1:
    age = st.number_input("Age", min_value=18, max_value=90, step=1, key="field_age")

with row1_col2:
    income = st.number_input("Annual Income (₹)", min_value=10000, max_value=50000000, step=25000, key="field_income")

with row1_col3:
    residence_type = st.selectbox("Residence Type", options=["Owned", "Rented", "Mortgage"], key="field_residence_type")

# Row 2: Loan Amount, Purpose & Type
row2_col1, row2_col2, row2_col3 = st.columns(3)
with row2_col1:
    loan_amount = st.number_input("Loan Amount (₹)", min_value=10000, max_value=25000000, step=10000, key="field_loan_amount")

with row2_col2:
    loan_purpose = st.selectbox("Loan Purpose", options=["Personal", "Home", "Education", "Auto"], key="field_loan_purpose")

with row2_col3:
    loan_type = st.selectbox("Loan Type", options=["Secured", "Unsecured"], key="field_loan_type")

# Row 3: Loan Tenure, Credit Utilization & Open Accounts
row3_col1, row3_col2, row3_col3 = st.columns(3)
with row3_col1:
    loan_tenure_months = st.number_input("Loan Tenure (Months)", min_value=6, max_value=120, step=1, key="field_loan_tenure_months")

with row3_col2:
    credit_utilization_ratio = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, step=1.0, key="field_credit_utilization_ratio")

with row3_col3:
    number_of_open_accounts = st.number_input("Number of Open Accounts", min_value=1, max_value=4, step=1, key="field_number_of_open_accounts")

# Row 4: Past Loan History & Delinquencies
row4_col1, row4_col2, row4_col3 = st.columns(3)
with row4_col1:
    total_loan_months = st.number_input("Total Loan History (Months)", min_value=1, max_value=120, step=1, key="field_total_loan_months")

with row4_col2:
    delinquent_months = st.number_input("Delinquent Months", min_value=0, max_value=60, step=1, key="field_delinquent_months")

with row4_col3:
    total_dpd = st.number_input("Total Days Past Due (DPD)", min_value=0, max_value=365, step=1, key="field_total_dpd")

st.write("")

# ----------------- Predict Action -----------------
calc_clicked = st.button("Calculate Credit Score", type="primary", use_container_width=True)

should_predict = calc_clicked or st.session_state["trigger_prediction"]

if should_predict:
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

        # Visual Credit Score Spectrum Bar
        score_val = result['credit_score']
        gauge_pct = min(max((score_val - 300) / (850 - 300) * 100, 0), 100)

        st.markdown(f"""
        <div class="gauge-wrapper">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 12px; font-weight: 700; color: #475569;">CREDIT SCORE SPECTRUM</span>
                <span style="font-size: 13px; font-weight: 800; color: {result['status_color']};">Score: {score_val} / 850</span>
            </div>
            <div class="gauge-bar-bg">
                <div class="gauge-pointer" style="left: {gauge_pct}%;"></div>
            </div>
            <div class="gauge-labels">
                <span>300 (Poor)</span>
                <span>465 (Subprime)</span>
                <span>575 (Fair)</span>
                <span>685 (Good)</span>
                <span>850 (Prime)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

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

# ----------------- Benchmark Matrix Expander -----------------
st.write("")
with st.expander("📋 View & Compare All Test Scenarios (Benchmark Matrix)", expanded=False):
    st.markdown("Below is a side-by-side comparison of the 5 standard test cases across credit grades, financial metrics, and model outputs.")

    summary_rows = []
    for key, tc in TEST_CASES.items():
        d = tc["data"]
        summary_rows.append({
            "Scenario": key,
            "Age": d["age"],
            "Income": f"₹{d['income']:,}",
            "Loan Amount": f"₹{d['loan_amount']:,}",
            "Tenure": f"{d['loan_tenure_months']}m",
            "Utilization": f"{d['credit_utilization_ratio']}%",
            "Delinquencies": f"{d['delinquent_months']}m",
            "Total DPD": f"{d['total_dpd']}d",
            "Predicted Score": tc["expected_score"],
            "Rating Grade": tc["expected_rating"],
            "Default Risk": tc["expected_prob"],
            "Underwriting Decision": tc["expected_decision"]
        })

    df_matrix = pd.DataFrame(summary_rows)
    st.dataframe(df_matrix, use_container_width=True, hide_index=True)
