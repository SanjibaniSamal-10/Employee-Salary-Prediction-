import joblib
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
# ------------------ Streamlit App ------------------
st.title("Employee Salary Prediction")
st.write("Predict whether an employee earns more than 50K/year based on their profile.")

# Load the trained model
model = joblib.load("salary_model.pkl")

# Sample category lists (same as used during training)
workclass_list = ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Federal-gov', 'Self-emp-inc', 'Without-pay']
education_list = ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', 'Assoc-voc', 'Doctorate', '5th-6th']
marital_status_list = ['Never-married', 'Married-civ-spouse', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent']
occupation_list = ['Tech-support', 'Craft-repair', 'Machine-op-inspct','Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners']
race_list = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
country_list = ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada']

# User Inputs
age = st.number_input("Age", min_value=18, max_value=90, value=30)
workclass = st.selectbox("Workclass", workclass_list)
education = st.selectbox("Education", education_list)
marital_status = st.selectbox("Marital Status", marital_status_list)
occupation = st.selectbox("Occupation", occupation_list)
race = st.selectbox("Race", race_list)
gender = st.selectbox("Gender", ['Male', 'Female'])
capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.slider("Hours per Week", 1, 100, 40)
native_country = st.selectbox("Native Country", country_list)

# Create input DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'education': [education],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
})

# Predict
if st.button("Predict Salary"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    if prediction == '>50K':
        st.success(f"The employee is likely to earn >50K (Confidence: {proba[1]*100:.2f}%)")
        st.progress(min(proba[1], 1.0))
    else:
        st.warning(f"The employee is likely to earn <=50K (Confidence: {proba[0]*100:.2f}%)")
        st.progress(min(proba[0], 1.0))
# --- Bar Chart of Prediction Probabilities ---
    st.markdown("### Prediction Confidence")
    fig, ax = plt.subplots()
    ax.bar(['<=50K', '>50K'], proba, color=['orange', 'green'])
    ax.set_ylabel("Probability")
    st.pyplot(fig)
# Batch predictions
st.markdown("---")
st.markdown("####  Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")
if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())
    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.write(batch_data.head())
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
