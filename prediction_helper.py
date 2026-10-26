import os
import joblib
import numpy as np
import pandas as pd

# Path to artifacts directory
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.joblib")

_model = None
_scaler = None
_cols_to_scale = None
_features = None


def load_artifacts():
    """
    Loads model and scaler objects from the artifacts directory.
    Supports combined model.joblib or separate model.joblib and scaler.joblib.
    """
    global _model, _scaler, _cols_to_scale, _features

    if _model is not None and _scaler is not None:
        return _model, _scaler, _cols_to_scale, _features

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    data = joblib.load(MODEL_PATH)

    if isinstance(data, dict):
        _model = data.get("model")
        _scaler = data.get("scaler")
        _cols_to_scale = list(data.get("cols_to_scale", []))
        _features = list(data.get("features", []))
    else:
        _model = data

    # Check for standalone scaler if not found in dict
    if _scaler is None and os.path.exists(SCALER_PATH):
        _scaler = joblib.load(SCALER_PATH)

    if _cols_to_scale is None and hasattr(_scaler, "feature_names_in_"):
        _cols_to_scale = list(_scaler.feature_names_in_)

    if _features is None:
        _features = [
            'number_of_open_accounts',
            'credit_utilization_ratio',
            'age',
            'loan_tenure_months',
            'loan_to_income',
            'delinquency_ratio',
            'avg_dpd_per_delinquency',
            'credit_utilization_per_income',
            'residence_type_Owned',
            'residence_type_Rented',
            'loan_purpose_Education',
            'loan_purpose_Home',
            'loan_purpose_Personal',
            'loan_type_Unsecured'
        ]

    return _model, _scaler, _cols_to_scale, _features


def calculate_derived_features(loan_amount, income, total_loan_months, delinquent_months, total_dpd, credit_utilization_ratio):
    """
    Calculates engineered features based on raw inputs:
    - loan_to_income: loan_amount / income
    - delinquency_ratio: delinquent_months / total_loan_months
    - avg_dpd_per_delinquency: total_dpd / delinquent_months
    - credit_utilization_per_income: credit_utilization_ratio / loan_to_income
    """
    income = float(income)
    loan_amount = float(loan_amount)
    total_loan_months = float(total_loan_months)
    delinquent_months = float(delinquent_months)
    total_dpd = float(total_dpd)
    credit_utilization_ratio = float(credit_utilization_ratio)

    loan_to_income = round(loan_amount / income, 2) if income > 0 else 0.0
    delinquency_ratio = round(delinquent_months / total_loan_months, 2) if total_loan_months > 0 else 0.0
    avg_dpd_per_delinquency = round(total_dpd / delinquent_months, 2) if delinquent_months > 0 else 0.0
    credit_utilization_per_income = round(credit_utilization_ratio / loan_to_income, 1) if loan_to_income > 0 else 0.0

    return {
        "loan_to_income": loan_to_income,
        "delinquency_ratio": delinquency_ratio,
        "avg_dpd_per_delinquency": avg_dpd_per_delinquency,
        "credit_utilization_per_income": credit_utilization_per_income
    }


def preprocess_inputs(
    age,
    income,
    loan_amount,
    loan_tenure_months,
    credit_utilization_ratio,
    number_of_open_accounts,
    total_loan_months,
    delinquent_months,
    total_dpd,
    residence_type,
    loan_purpose,
    loan_type
):
    """
    Prepares input vector, scales continuous features, and encodes categorical features.
    """
    model, scaler, cols_to_scale, features = load_artifacts()

    # Derived metrics
    derived = calculate_derived_features(
        loan_amount=loan_amount,
        income=income,
        total_loan_months=total_loan_months,
        delinquent_months=delinquent_months,
        total_dpd=total_dpd,
        credit_utilization_ratio=credit_utilization_ratio
    )

    # Build DataFrame for scaler matching cols_to_scale
    scaler_row = {col: 0.0 for col in cols_to_scale}
    scaler_row["number_of_open_accounts"] = float(number_of_open_accounts)
    scaler_row["credit_utilization_ratio"] = float(credit_utilization_ratio)
    scaler_row["age"] = float(age)
    scaler_row["loan_tenure_months"] = float(loan_tenure_months)
    scaler_row["loan_to_income"] = derived["loan_to_income"]
    scaler_row["delinquency_ratio"] = derived["delinquency_ratio"]
    scaler_row["avg_dpd_per_delinquency"] = derived["avg_dpd_per_delinquency"]
    scaler_row["credit_utilization_per_income"] = derived["credit_utilization_per_income"]

    df_scaler_input = pd.DataFrame([scaler_row], columns=cols_to_scale)
    scaled_array = scaler.transform(df_scaler_input)
    df_scaled = pd.DataFrame(scaled_array, columns=cols_to_scale)

    # Build final feature vector matching model features exactly
    final_row = {feat: 0.0 for feat in features}

    continuous_features = [
        "number_of_open_accounts",
        "credit_utilization_ratio",
        "age",
        "loan_tenure_months",
        "loan_to_income",
        "delinquency_ratio",
        "avg_dpd_per_delinquency",
        "credit_utilization_per_income"
    ]

    for col in continuous_features:
        if col in features and col in df_scaled.columns:
            final_row[col] = df_scaled[col].iloc[0]

    # One-hot encoded categorical variables
    residence = str(residence_type).strip().title()
    if residence == "Owned":
        final_row["residence_type_Owned"] = 1.0
    elif residence == "Rented":
        final_row["residence_type_Rented"] = 1.0
    # 'Mortgage' is the baseline (0 for both Owned and Rented)

    purpose = str(loan_purpose).strip().title()
    if purpose == "Education":
        final_row["loan_purpose_Education"] = 1.0
    elif purpose == "Home":
        final_row["loan_purpose_Home"] = 1.0
    elif purpose in ["Personal", "Personaal"]:
        final_row["loan_purpose_Personal"] = 1.0
    # 'Auto' is the baseline (0 for Education, Home, Personal)

    ltype = str(loan_type).strip().title()
    if ltype == "Unsecured":
        final_row["loan_type_Unsecured"] = 1.0
    # 'Secured' is the baseline (0)

    final_df = pd.DataFrame([final_row], columns=features)
    return final_df, derived


def predict(
    age,
    income,
    loan_amount,
    loan_tenure_months,
    credit_utilization_ratio,
    number_of_open_accounts,
    total_loan_months,
    delinquent_months,
    total_dpd,
    residence_type,
    loan_purpose,
    loan_type
):
    """
    Main prediction function.
    Returns default probability, credit score (300-850), risk rating, and recommendation.
    """
    model, _, _, _ = load_artifacts()

    final_df, derived = preprocess_inputs(
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

    # Probabilities and predictions
    prob_array = model.predict_proba(final_df)[0]
    default_prob = float(prob_array[1])
    prediction = int(model.predict(final_df)[0])

    # Credit score computation (Scale 300 to 850)
    credit_score = int(np.clip(round(300 + (1.0 - default_prob) * 550), 300, 850))

    # Risk tiers and recommendations
    if default_prob < 0.15:
        risk_tier = "Very Low Risk"
        rating = "Prime (AAA)"
        recommendation = "Approved - Prime Terms"
        status_color = "#10b981"  # Emerald green
    elif default_prob < 0.30:
        risk_tier = "Low Risk"
        rating = "Good (AA)"
        recommendation = "Approved - Standard Rate"
        status_color = "#34d399"  # Light green
    elif default_prob < 0.50:
        risk_tier = "Moderate Risk"
        rating = "Fair (BBB)"
        recommendation = "Conditional Approval / Manual Review"
        status_color = "#f59e0b"  # Amber
    elif default_prob < 0.70:
        risk_tier = "High Risk"
        rating = "Subprime (B)"
        recommendation = "Review / Additional Collateral Required"
        status_color = "#f97316"  # Orange
    else:
        risk_tier = "Very High Risk"
        rating = "Poor (D)"
        recommendation = "Decline - High Probability of Default"
        status_color = "#ef4444"  # Red

    return {
        "probability": default_prob,
        "probability_pct": round(default_prob * 100, 2),
        "credit_score": credit_score,
        "prediction": prediction,
        "risk_tier": risk_tier,
        "rating": rating,
        "recommendation": recommendation,
        "status_color": status_color,
        "loan_to_income": derived["loan_to_income"],
        "delinquency_ratio": derived["delinquency_ratio"],
        "avg_dpd_per_delinquency": derived["avg_dpd_per_delinquency"],
        "credit_utilization_per_income": derived["credit_utilization_per_income"],
        "feature_df": final_df
    }
