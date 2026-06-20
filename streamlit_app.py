import os
import streamlit as st
import pandas as pd
import joblib
from model import get_or_train_model

# Page configurations
st.set_page_config(page_title="Stroke Risk Predictor", layout="centered", page_icon="🧠")

st.title("🧠 Stroke Prediction")
st.write("This app uses a Machine Learning model trained on clinical and lifestyle data to predict the likelihood of experiencing a stroke.")

# Load or auto-train the model pipeline in the background
@st.cache_resource
def load_cached_model():
    return get_or_train_model()

try:
    with st.spinner("Initializing dataset and predictive model..."):
        model_pipeline = load_cached_model()
    st.success("System Ready for Assessments!")
except Exception as e:
    st.error(f"Failed to load/train model. Error logs: {e}")
    st.stop()

st.markdown("---")
st.subheader("📋 Enter Patient Clinical Information")

# Streamlit form inputs matching original columns:
# ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
        age = st.number_input("Age", min_value=0.0, max_value=120.0, value=45.0)
        hypertension = st.selectbox("Hypertension (High Blood Pressure)?", options=["No", "Yes"])
        heart_disease = st.selectbox("History of Heart Disease?", options=["No", "Yes"])
        ever_married = st.selectbox("Ever Married?", options=["Yes", "No"])

    with col2:
        work_type = st.selectbox("Employment Sector", options=["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence_type = st.selectbox("Residence Environment", options=["Urban", "Rural"])
        avg_glucose_level = st.number_input("Average Blood Glucose Level (mg/dL)", min_value=40.0, max_value=300.0, value=95.0)
        bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=70.0, value=25.0)
        smoking_status = st.selectbox("Smoking Profile", options=["never smoked", "formerly smoked", "smokes", "Unknown"])

    # Fixed line here:
    submit_button = st.form_submit_button("Analyze Stroke Risk")


if submit_button:
    # 1. Transform user input to match the structure of the training dataset
    input_data = {
        'gender': gender,
        'age': age,
        'hypertension': 1 if hypertension == "Yes" else 0,
        'heart_disease': 1 if heart_disease == "Yes" else 0,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }
    
    # Convert input dict to Pandas DataFrame for pipeline handling
    input_df = pd.DataFrame([input_data])
    
    # 2. Extract prediction probabilities and classes from pipeline
    prediction = model_pipeline.predict(input_df)[0]
    prediction_proba = model_pipeline.predict_proba(input_df)[0][1] # Probability of stroke (class 1)
    
    st.markdown("---")
    st.subheader("📊 Diagnostic Results")
    
    # Display results safely
    if prediction == 1:
        st.error(f"⚠️ **High Risk Status:** The model detects a high risk profile for a stroke event. (Risk Probability: {prediction_proba * 100:.1f}%)")
        st.write("Please consult a healthcare professional regarding cardiovascular health and management vectors.")
    else:
        st.success(f"✅ **Low Risk Status:** The model detects a low risk profile for a stroke event. (Risk Probability: {prediction_proba * 100:.1f}%)")
        st.write("Maintain healthy lifestyle choices to keep clinical stats within optimal ranges.")
