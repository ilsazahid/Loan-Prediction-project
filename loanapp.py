
import streamlit as st
import pickle
import numpy as np

# ============================================================
# LOAD MODEL
# ============================================================

with open("/content/loan_model.pkl", "rb") as f:
    model = pickle.load(f)

# ============================================================
# LOAD SCALER
# ============================================================

with open("/content/loan_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ============================================================
# TITLE
# ============================================================

st.title("🏦 Loan Approval Prediction App")

st.write("Enter Applicant Details")

# ============================================================
# USER INPUTS
# ============================================================

Gender = st.selectbox(
    "Gender",
    [0, 1]
)

Married = st.selectbox(
    "Married",
    [0, 1]
)

Dependents = st.selectbox(
    "Dependents",
    [0, 1, 2, 3]
)

Education = st.selectbox(
    "Education",
    [0, 1]
)

Self_Employed = st.selectbox(
    "Self Employed",
    [0, 1]
)

ApplicantIncome = st.number_input(
    "Applicant Income",
    0,
    100000
)

CoapplicantIncome = st.number_input(
    "Coapplicant Income",
    0,
    100000
)

LoanAmount = st.number_input(
    "Loan Amount",
    0,
    1000
)

Loan_Amount_Term = st.number_input(
    "Loan Amount Term",
    0,
    500
)

Credit_History = st.selectbox(
    "Credit History",
    [0, 1]
)

Property_Area = st.selectbox(
    "Property Area",
    [0, 1, 2]
)

# ============================================================
# PREPARE INPUT DATA
# ============================================================

features = np.array([[
    Gender,
    Married,
    Dependents,
    Education,
    Self_Employed,
    ApplicantIncome,
    CoapplicantIncome,
    LoanAmount,
    Loan_Amount_Term,
    Credit_History,
    Property_Area
]])

# ============================================================
# SCALE INPUT
# ============================================================

features = scaler.transform(features)

# ============================================================
# PREDICTION
# ============================================================

if st.button("Predict"):

    result = model.predict(features)

    if result[0] == 1:
        st.success("✅ Loan Approved")

    else:
        st.error("❌ Loan Not Approved")
