import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Loan Approval Predictor")

@st.cache_resource
def load_files():
    model   = joblib.load("loan_model.pkl")
    num_imp = joblib.load("num_imputer.pkl")
    cat_imp = joblib.load("cat_imputer.pkl")
    scaler  = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, num_imp, cat_imp, scaler, columns

model, num_imp, cat_imp, scaler, columns = load_files()

num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
cat_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Credit_History"]

st.title("Loan Approval Predictor")
st.write("Enter the applicant details to see if the loan is likely to be approved.")

credit_history = st.selectbox("Credit History", ["Good", "Bad"])
income = st.number_input("Applicant Income (per month)", min_value=0.0, value=5000.0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0.0, value=130.0)
married = st.selectbox("Married", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.button("Predict"):
    row = pd.DataFrame(np.nan, index=[0], columns=columns)

    row["Credit_History"]  = 1.0 if credit_history == "Good" else 0.0
    row["ApplicantIncome"] = income
    row["LoanAmount"]      = loan_amount
    row["Married"]         = 1.0 if married == "Yes" else 0.0

    row["Property_Area_Rural"]     = 1 if property_area == "Rural" else 0
    row["Property_Area_Semiurban"] = 1 if property_area == "Semiurban" else 0
    row["Property_Area_Urban"]     = 1 if property_area == "Urban" else 0

    row[num_cols] = num_imp.transform(row[num_cols])
    row[cat_cols] = cat_imp.transform(row[cat_cols])
    row[num_cols] = scaler.transform(row[num_cols])
    row = row[columns]

    prediction = model.predict(row)[0]
    prob = model.predict_proba(row)[0][1]

    if prediction == 1:
        st.success(f"✅ Loan likely APPROVED (confidence: {prob:.0%})")
    else:
        st.error(f"❌ Loan likely REJECTED (approval chance: {prob:.0%})")