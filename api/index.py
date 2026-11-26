
from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

app = Flask(__name__, template_folder="../templates")

# Get project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load trained ML files
model = joblib.load(BASE_DIR / "loan_model.pkl")
num_imp = joblib.load(BASE_DIR / "num_imputer.pkl")
cat_imp = joblib.load(BASE_DIR / "cat_imputer.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
columns = joblib.load(BASE_DIR / "columns.pkl")

num_cols = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term"
]

cat_cols = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Credit_History"
]



@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probability = None

    # Default values
    income = 5000
    loan_amount = 130
    credit_history = "Good"
    married = "Yes"
    property_area = "Urban"

    if request.method == "POST":

        credit_history = request.form["credit_history"]
        income = float(request.form["income"])
        loan_amount = float(request.form["loan_amount"])
        married = request.form["married"]
        property_area = request.form["property_area"]

        row = pd.DataFrame(
            np.nan, index=[0], columns=columns
        )

        row["Credit_History"] = 1.0 if credit_history == "Good" else 0.0
        row["ApplicantIncome"] = income
        row["LoanAmount"] = loan_amount
        row["Married"] = 1.0 if married == "Yes" else 0.0

        row["Property_Area_Rural"] = 1 if property_area == "Rural" else 0
        row["Property_Area_Semiurban"] = 1 if property_area == "Semiurban" else 0
        row["Property_Area_Urban"] = 1 if property_area == "Urban" else 0

        row[num_cols] = num_imp.transform(row[num_cols])
        row[cat_cols] = cat_imp.transform(row[cat_cols])
        row[num_cols] = scaler.transform(row[num_cols])
        row = row[columns]

        result = model.predict(row)[0]
        probability = model.predict_proba(row)[0][1]

        prediction = "APPROVED" if result == 1 else "REJECTED"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        income=income,
        loan_amount=loan_amount,
        credit_history=credit_history,
        married=married,
        property_area=property_area
    )

if __name__ == "__main__":
    app.run(debug=True)
