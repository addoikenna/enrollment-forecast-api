from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from forecast import forecast_next_12_months

app = FastAPI(
    title="Enrollment Forecast API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Enrollment Forecast API is running"
    }


@app.get("/forecast")
def get_forecast():
    forecast_df = forecast_next_12_months()

    forecast_df["month"] = forecast_df["month"].astype(str)

    return {
        "forecast_horizon": "next_12_months",
        "data": forecast_df.to_dict(orient="records")
    }


@app.get("/model-info")
def model_info():
    return {
        "model": "Ridge Regression",
        "forecast_type": "12-month enrollment forecast",
        "features": [
            "month_sin",
            "month_cos",
            "cohort_stage",
            "is_cohort_close_month",
            "lagged enrollments",
            "lagged accepted applications"
        ]
    }
