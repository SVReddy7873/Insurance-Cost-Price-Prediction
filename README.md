# Health Insurance Cost Prediction using Machine Learning

A robust Random Forest regression project that predicts annual medical insurance charges from demographic and health-related features.

## Project name for resume
**Health Insurance Cost Prediction**

## Expected dataset
Place your `insurance.csv` at:

`data/insurance.csv`

Expected columns:

`age, sex, bmi, children, smoker, region, charges`

The trainer also tolerates common column-name variations.

## Run

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
python train.py --data data/insurance.csv
streamlit run app.py
```

## What makes this version stronger

- End-to-end sklearn Pipeline prevents preprocessing mismatch between training and prediction.
- Missing numeric values use median imputation.
- Missing categorical values use most-frequent imputation.
- One-hot encoding uses `handle_unknown='ignore'`, so unseen categories do not crash prediction.
- Input validation prevents invalid UI values.
- Random Forest is persisted with joblib.
- Metrics are saved in JSON.
- The application fails gracefully when the model is missing or prediction encounters an unexpected problem.
- No data leakage from test data into preprocessing.
- MAE, MSE, RMSE and R² are calculated.

## Important limitation

No ML program can honestly be guaranteed to never fail under every possible environment. A missing Python installation, corrupted dataset, incompatible library version, disk failure, or deliberately malformed file can still prevent execution. This project validates expected inputs and handles normal failure cases explicitly.

## Resume bullets

**Health Insurance Cost Prediction using Machine Learning**
- Built a Random Forest regression pipeline to estimate medical insurance charges from demographic and health-related features.
- Implemented robust preprocessing with missing-value handling, categorical encoding, feature scaling, and leakage-safe train-test evaluation.
- Evaluated performance using MAE, MSE, RMSE and R², and deployed the serialized pipeline through an interactive Streamlit prediction interface.
