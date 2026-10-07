import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("bank_marketing_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="centered"
)

# Title
st.title("🏦 Bank Marketing Prediction")

st.write(
    "This application predicts whether a customer is likely "
    "to accept a term deposit offer."
)

st.divider()

# Customer Information
st.header("Customer Information")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

job = st.selectbox(
    "Job",
    [
        "admin.", "blue-collar", "entrepreneur", "housemaid",
        "management", "retired", "self-employed", "services",
        "student", "technician", "unemployed", "unknown"
    ]
)

marital = st.selectbox(
    "Marital Status",
    ["married", "single", "divorced", "unknown"]
)

education = st.selectbox(
    "Education",
    [
        "basic.4y", "basic.6y", "basic.9y",
        "high.school", "illiterate",
        "professional.course", "university.degree",
        "unknown"
    ]
)

default = st.selectbox(
    "Credit Default",
    ["no", "yes", "unknown"]
)

housing = st.selectbox(
    "Housing Loan",
    ["no", "yes", "unknown"]
)

loan = st.selectbox(
    "Personal Loan",
    ["no", "yes", "unknown"]
)

# Campaign Information
st.header("Campaign Information")

contact = st.selectbox(
    "Contact Method",
    ["cellular", "telephone"]
)

month = st.selectbox(
    "Month",
    [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ]
)

day_of_week = st.selectbox(
    "Day of Week",
    ["mon", "tue", "wed", "thu", "fri"]
)

campaign = st.number_input(
    "Number of Contacts in Current Campaign",
    min_value=1,
    value=1
)

pdays = st.number_input(
    "Days Since Previous Contact",
    min_value=0,
    value=999
)

previous = st.number_input(
    "Number of Previous Contacts",
    min_value=0,
    value=0
)

poutcome = st.selectbox(
    "Previous Campaign Outcome",
    ["failure", "nonexistent", "success"]
)

# Economic Information
st.header("Economic Information")

emp_var_rate = st.number_input(
    "Employment Variation Rate",
    value=1.1
)

cons_price_idx = st.number_input(
    "Consumer Price Index",
    value=93.994
)

cons_conf_idx = st.number_input(
    "Consumer Confidence Index",
    value=-36.4
)

euribor3m = st.number_input(
    "Euribor 3 Month Rate",
    value=4.857
)

nr_employed = st.number_input(
    "Number of Employees",
    value=5191.0
)

st.divider()

# Prediction Button
if st.button("🔍 Predict", use_container_width=True):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "age": [age],
        "job": [job],
        "marital": [marital],
        "education": [education],
        "default": [default],
        "housing": [housing],
        "loan": [loan],
        "contact": [contact],
        "month": [month],
        "day_of_week": [day_of_week],
        "campaign": [campaign],
        "pdays": [pdays],
        "previous": [previous],
        "poutcome": [poutcome],
        "emp.var.rate": [emp_var_rate],
        "cons.price.idx": [cons_price_idx],
        "cons.conf.idx": [cons_conf_idx],
        "euribor3m": [euribor3m],
        "nr.employed": [nr_employed]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display result
    st.subheader("Prediction Result")

    if prediction == "yes":
        st.success(
            "✅ The customer is likely to accept the term deposit offer."
        )
    else:
        st.warning(
            "❌ The customer is unlikely to accept the term deposit offer."
        )