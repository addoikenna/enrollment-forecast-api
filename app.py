from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query

from forecast import (
    forecast_next_6_months,
    get_current_year_projection,
    get_daily_pace,
    save_daily_forecast_history,
    get_forecast_history_months,
    get_eligible_programs,
    get_forecast_overview,
    backfill_program_forecast_history,
    backfill_all_programs_forecast
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
def get_forecast(
    program: str = Query(default="All Programs")
):
    return get_forecast_overview(program=program)

    
@app.get("/daily-pace")
def daily_pace(
    month: str | None = Query(default=None),
    program: str = Query(default="All Programs")
):
    return get_daily_pace(
        month=month,
        program=program
    )


@app.get("/eligible-programs")
def eligible_programs():
    return get_eligible_programs()


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
    

@app.get("/save-daily-forecast")
def save_daily_forecast():
    return save_daily_forecast_history()

    
@app.get("/forecast-history-months")
def forecast_history_months():
    return get_forecast_history_months()


@app.get("/backfill-program-forecast")
def backfill_program_forecast(
    month: str = Query(default="2026-05-01")
):
    return backfill_program_forecast_history(month=month)


# -----------------------------------------------------
@app.get("/backfill-all-programs-forecast")
def backfill_all_programs(
    month: str = Query(default="2026-06-01")
):
    return backfill_all_programs_forecast(month=month)
