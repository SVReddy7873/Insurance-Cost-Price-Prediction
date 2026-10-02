
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "insurance_cost_pipeline.joblib"
METRICS_PATH = BASE / "models" / "metrics.json"

st.set_page_config(page_title="Health Insurance Cost Predictor", page_icon="💳", layout="centered")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

def validate_inputs(age, sex, bmi, children, smoker, region):
    errors = []
    if not 0 <= age <= 120:
        errors.append("Age must be between 0 and 120.")
    if not 5 <= bmi <= 100:
        errors.append("BMI must be between 5 and 100.")
    if not 0 <= children <= 30:
        errors.append("Number of children must be between 0 and 30.")
    if sex not in {"male", "female"}:
        errors.append("Please select a valid sex.")
    if smoker not in {"yes", "no"}:
        errors.append("Please select a valid smoking status.")
    if region not in {"northeast", "northwest", "southeast", "southwest"}:
        errors.append("Please select a valid region.")
    return errors

def main():
    st.title("💳 Health Insurance Cost Predictor")
    st.caption("Random Forest regression • validated inputs • saved preprocessing + model pipeline")

    model = load_model()
    metrics = load_metrics()

    if model is None:
        st.error("Trained model not found. Run: python train.py --data data/insurance.csv")
        st.stop()

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
            bmi = st.number_input("BMI", min_value=5.0, max_value=100.0, value=27.5, step=0.1)
            children = st.number_input("Children", min_value=0, max_value=30, value=0, step=1)
        with col2:
            sex = st.selectbox("Sex", ["male", "female"])
            smoker = st.selectbox("Smoker", ["no", "yes"])
            region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

        submitted = st.form_submit_button("Predict Insurance Cost", type="primary")

    if submitted:
        errors = validate_inputs(age, sex, bmi, children, smoker, region)
        if errors:
            for error in errors:
                st.error(error)
            return

        row = pd.DataFrame([{
            "age": age,
            "bmi": bmi,
            "children": children,
            "sex": sex,
            "smoker": smoker,
            "region": region,
        }])

        try:
            prediction = float(model.predict(row)[0])
            if not np.isfinite(prediction):
                raise ValueError("The model returned a non-finite prediction.")
            prediction = max(0.0, prediction)
            st.success(f"Estimated annual medical insurance charge: **${prediction:,.2f}**")
            st.info("This is a machine-learning estimate, not an insurance quote or medical/financial advice.")
        except Exception as exc:
            st.error("Prediction failed safely. Please verify the model file and inputs.")
            st.exception(exc)

    if metrics:
        st.divider()
        st.subheader("Model Performance")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"${metrics.get('mae', 0):,.2f}")
        c2.metric("RMSE", f"${metrics.get('rmse', 0):,.2f}")
        c3.metric("R²", f"{metrics.get('r2', 0):.3f}")

if __name__ == "__main__":
    main()
