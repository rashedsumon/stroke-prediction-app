# Stroke Prediction Web Application

An end-to-end Machine Learning web application deployed on Streamlit Cloud that predicts the likelihood of a patient suffering a stroke based on demographics, clinical metrics, and lifestyle attributes.

## Features
- **Auto-Dataset Download:** Uses `kagglehub` to download the live dataset dynamically.
- **Robust ML Pipeline:** Implements a Scikit-Learn Random Forest Classifier handling imputation and encoding automatically.
- **Interactive UI:** Built with Streamlit for seamless user data entry and instant predictions.

## Local Setup
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt