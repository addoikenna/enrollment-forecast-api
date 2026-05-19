from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from forecast import (
    forecast_next_6_months,
    get_current_year_projection,
    get_daily_pace
)


app = FastAPI(
    title="Enrollment Forecast API",
    version="1.1"
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
    forecast_df = forecast_next_6_months()

    forecast_df["month"] = forecast_df["month"].astype(str)

    return {
        "forecast_horizon": "next_6_months",
        "data": forecast_df.to_dict(orient="records"),
        "year_projection": get_current_year_projection()
    }


@app.get("/daily-pace")
def daily_pace():
    return get_daily_pace()


@app.get("/model-info")
def model_info():
    return {
        "model": "Ridge Regression",
        "forecast_type": "rolling 6-month enrollment forecast",
        "additional_outputs": [
            "current year projected enrollment",
            "daily enrollment pace tracking",
            "weekly forecast rollup"
        ],
        "features": [
            "month_sin",
            "month_cos",
            "cohort_stage",
            "is_cohort_close_month",
            "lagged enrollments",
            "lagged accepted applications"
        ]
    }
