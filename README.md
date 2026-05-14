# Enrollment Forecast API

An AI-powered enrollment forecasting system built using:

- Ridge Regression
- Cohort-cycle feature engineering
- FastAPI
- Recursive forecasting

The system generates real-time 6-month enrollment forecasts based on historical enrollment and accepted application data.

---
# Features

- Monthly enrollment forecasting
- Cohort-aware seasonality modeling
- Recursive future prediction
- REST API endpoints
- FastAPI backend
- Ready for dashboard integration (Lovable)

---
# Project Structure

```text
.
├── app.py
├── forecast.py
├── requirements.txt
├── models/
│   ├── ridge_enrollment_model.pkl
│   └── ridge_model_features.pkl
└── data/
    └── model_dataset.csv
```

---

# API Endpoints

## Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok",
  "message": "Enrollment Forecast API is running"
}
```

---

## Forecast Endpoint

```http
GET /forecast
```

Returns:

```json
{
  "forecast_horizon": "next_6_months",
  "data": [
    {
      "month": "2026-05-01",
      "forecasted_enrollments": 3047
    }
  ]
}
```

---

## Model Info

```http
GET /model-info
```

---

# Model Overview

The forecasting model uses:

- cohort-cycle logic
- seasonal encoding
- lagged enrollment behavior
- lagged accepted applications

The project discovered that enrollment behavior is driven more strongly by operational cohort cycles than by traditional statistical seasonality alone.

---

# Tech Stack

- Python
- FastAPI
- Scikit-learn
- Pandas
- NumPy
- Uvicorn

---

# Deployment

Designed for deployment on:

- Render
- Railway
- Google Cloud Run

---

# Future Improvements

- Ensemble forecasting
- Prophet integration
- Real-time Google Sheets ingestion
- Confidence intervals
- Automated retraining
- Dashboard analytics

---

# Author

Built as part of an AI-powered enrollment forecasting system project.
