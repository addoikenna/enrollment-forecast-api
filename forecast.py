import pandas as pd
import numpy as np
import joblib
import requests


# ---------------------------------------
# Model assets
# ---------------------------------------

ridge_model = joblib.load("models/ridge_enrollment_model.pkl")
features = joblib.load("models/ridge_model_features.pkl")


# ---------------------------------------
# Google Sheet IDs and GIDs
# ---------------------------------------

ENROLLMENT_SHEET_ID = "1bkUlCdL0VpCy-2Gm17MfTlC2lbv-qQoNQbasN2cGniw"
APPLICATIONS_SHEET_ID = "1ry6T6I0qHbme3Bb1yVTSZyop9Cn1ictFU93V-lOlrfo"
FORECAST_HISTORY_WEB_APP_URL = "https://script.google.com/macros/s/AKfycbyueag5B7rLm6zxy2g88olVH8R_ftGI9ZEAKBJ_mmnJFLRSXY5nnQbYm229QDLmgKT4jg/exec"

MODEL_VERSION = "Ridge_v1"

ENROLLMENT_GID = "1781196956"
APPLICATIONS_GID = "3304352"
DAILY_ENROLLMENT_GID = "604024081"


# ---------------------------------------
# Helpers
# ---------------------------------------

def google_sheet_csv_url(sheet_id, gid):
    return (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}"
        f"/export?format=csv&gid={gid}"
    )


def convert_month_column(value):
    if pd.isna(value):
        return pd.NaT

    try:
        numeric_value = float(value)

        if numeric_value > 10000:
            return (
                pd.to_datetime("1899-12-30")
                + pd.to_timedelta(numeric_value, unit="D")
            )
    except Exception:
        pass

    return pd.to_datetime(value)


def create_cohort_stage(month):
    stage_mapping = {
        10: 1, 11: 2, 12: 3, 1: 4,
        2: 1, 3: 2, 4: 3, 5: 4,
        6: 1, 7: 2, 8: 3, 9: 4,
    }

    return stage_mapping[month]


def format_period(start_month, end_month):
    if start_month is None or end_month is None:
        return "None"

    return (
        f"{pd.Timestamp(start_month).strftime('%b %Y')} - "
        f"{pd.Timestamp(end_month).strftime('%b %Y')}"
    )


def to_records(df):
    return df.to_dict(orient="records")


# ---------------------------------------
# Load live monthly Google Sheet data
# ---------------------------------------

def load_live_data():
    enrollment_url = google_sheet_csv_url(
        ENROLLMENT_SHEET_ID,
        ENROLLMENT_GID
    )

    applications_url = google_sheet_csv_url(
        APPLICATIONS_SHEET_ID,
        APPLICATIONS_GID
    )

    enroll_df = pd.read_csv(enrollment_url)
    apps_df = pd.read_csv(applications_url)

    enroll_df.columns = enroll_df.columns.str.strip().str.lower()
    apps_df.columns = apps_df.columns.str.strip().str.lower()

    enroll_df = enroll_df.rename(columns={
        "enrolment_month": "month",
        "enrollment_month": "month",
    })

    apps_df = apps_df.rename(columns={
        "app_month": "month",
    })

    enroll_df["month"] = enroll_df["month"].apply(convert_month_column)
    apps_df["month"] = apps_df["month"].apply(convert_month_column)

    enroll_df = enroll_df[[
        "month",
        "enrolments",
        "programs"
    ]]

    apps_df = apps_df[[
        "month",
        "applications_submitted",
        "applications_accepted"
    ]]

    df = pd.merge(
        enroll_df,
        apps_df,
        on="month",
        how="inner"
    )

    df = df.rename(columns={
        "enrolments": "enrollments"
    })

    df = df.sort_values("month").reset_index(drop=True)

    # Remove launch month distortion
    df = df[df["month"] != pd.Timestamp("2023-05-01")]
    df = df.reset_index(drop=True)

    return df


# ---------------------------------------
# Load live daily enrollment data
# ---------------------------------------

def load_daily_enrollment_data():
    daily_url = google_sheet_csv_url(
        ENROLLMENT_SHEET_ID,
        DAILY_ENROLLMENT_GID
    )

    daily_df = pd.read_csv(daily_url)

    daily_df.columns = daily_df.columns.str.strip().str.lower()

    daily_df = daily_df.rename(columns={
        "enrolment_date": "date",
        "daily_enrolments": "enrollments"
    })

    daily_df["date"] = daily_df["date"].apply(convert_month_column)

    daily_df = daily_df[[
        "date",
        "enrollments"
    ]]

    daily_df = daily_df.sort_values("date").reset_index(drop=True)

    return daily_df


# ---------------------------------------
# Monthly forecast functions
# ---------------------------------------

def forecast_next_months(periods=6):
    forecast_df = load_live_data()
    predictions = []

    future_months = pd.date_range(
        start=forecast_df["month"].max() + pd.offsets.MonthBegin(1),
        periods=periods,
        freq="MS"
    )

    for future_month in future_months:
        month_number = future_month.month
        row = {}

        row["month_sin"] = np.sin(2 * np.pi * month_number / 12)
        row["month_cos"] = np.cos(2 * np.pi * month_number / 12)

        row["is_cohort_close_month"] = int(month_number in [1, 5, 9])
        row["cohort_stage"] = create_cohort_stage(month_number)

        row["enrollments_lag_1"] = forecast_df["enrollments"].iloc[-1]
        row["enrollments_lag_2"] = forecast_df["enrollments"].iloc[-2]
        row["enrollments_lag_12"] = forecast_df["enrollments"].iloc[-12]

        row["apps_accepted_lag_1"] = (
            forecast_df["applications_accepted"].iloc[-1]
        )

        row["apps_accepted_lag_2"] = (
            forecast_df["applications_accepted"].iloc[-2]
        )

        row["apps_accepted_roll_3"] = (
            forecast_df["applications_accepted"].iloc[-3:].mean()
        )

        if "days_in_month" in features:
            row["days_in_month"] = future_month.days_in_month

        if "programs" in features:
            row["programs"] = forecast_df["programs"].iloc[-1]

        if "enrollments_roll_3" in features:
            row["enrollments_roll_3"] = (
                forecast_df["enrollments"].iloc[-3:].mean()
            )

        if "conversion_rate_lag_1" in features:
            row["conversion_rate_lag_1"] = (
                forecast_df["enrollments"].iloc[-1]
                / forecast_df["applications_accepted"].iloc[-1]
            )

        X_future = pd.DataFrame([row])[features]

        prediction = ridge_model.predict(X_future)[0]
        prediction = max(0, prediction)
        prediction = int(np.floor(prediction))

        predictions.append(prediction)

        new_row = forecast_df.iloc[-1].copy()
        new_row["month"] = future_month
        new_row["enrollments"] = prediction

        forecast_df = pd.concat(
            [forecast_df, pd.DataFrame([new_row])],
            ignore_index=True
        )

    forecast_output = pd.DataFrame({
        "month": future_months,
        "forecasted_enrollments": predictions
    })

    return forecast_output


def forecast_next_6_months():
    return forecast_next_months(periods=6)


def forecast_next_12_months():
    return forecast_next_months(periods=12)


# ---------------------------------------
# Current-year projection
# ---------------------------------------

def get_current_year_projection():
    df = load_live_data()

    today = pd.Timestamp.today()
    current_year = today.year
    current_month = today.month

    actual_df = df[
        (df["month"].dt.year == current_year)
        & (df["month"].dt.month < current_month)
    ]

    actual_closed_months_total = int(actual_df["enrollments"].sum())
    actual_months_count = int(len(actual_df))

    months_remaining = 12 - current_month + 1

    forecast_df = forecast_next_months(periods=months_remaining)

    forecast_df = forecast_df[
        forecast_df["month"].dt.year == current_year
    ]

    forecast_remaining_months_total = int(
        forecast_df["forecasted_enrollments"].sum()
    )

    forecast_months_count = int(len(forecast_df))

    projected_year_total = (
        actual_closed_months_total
        + forecast_remaining_months_total
    )

    if actual_months_count > 0:
        actual_start = actual_df["month"].min()
        actual_end = actual_df["month"].max()
    else:
        actual_start = None
        actual_end = None

    if forecast_months_count > 0:
        forecast_start = forecast_df["month"].min()
        forecast_end = forecast_df["month"].max()
    else:
        forecast_start = None
        forecast_end = None

    return {
        "year": int(current_year),
        "actual_closed_months_total": actual_closed_months_total,
        "forecast_remaining_months_total": forecast_remaining_months_total,
        "projected_year_total": int(projected_year_total),
        "actual_months_count": actual_months_count,
        "forecast_months_count": forecast_months_count,
        "actual_period": format_period(actual_start, actual_end),
        "forecast_period": format_period(forecast_start, forecast_end),
    }


# ---------------------------------------
# Daily weight profile
# ---------------------------------------

def build_daily_day_weight_profile():
    daily_df = load_daily_enrollment_data()

    daily_df["month_period"] = daily_df["date"].dt.to_period("M").astype(str)
    daily_df["day"] = daily_df["date"].dt.day
    daily_df["month"] = daily_df["date"].dt.month

    daily_df["is_cohort_close_month"] = daily_df["month"].isin([1, 5, 9])

    daily_df["month_type"] = np.where(
        daily_df["is_cohort_close_month"],
        "Cohort Close Month",
        "Normal Month"
    )

    monthly_totals = (
        daily_df
        .groupby("month_period")["enrollments"]
        .sum()
        .reset_index()
        .rename(columns={"enrollments": "monthly_total"})
    )

    daily_df = daily_df.merge(
        monthly_totals,
        on="month_period",
        how="left"
    )

    daily_df["daily_weight"] = (
        daily_df["enrollments"] / daily_df["monthly_total"]
    )

    daily_day_weight_profile = (
        daily_df
        .groupby(["month_type", "day"])["daily_weight"]
        .mean()
        .reset_index()
    )

    daily_day_weight_profile["normalized_weight"] = (
        daily_day_weight_profile
        .groupby("month_type")["daily_weight"]
        .transform(lambda x: x / x.sum())
    )

    return daily_day_weight_profile


# ---------------------------------------
# Daily forecast allocation engine
# ---------------------------------------

def create_daily_forecast_allocation(
    forecast_month,
    monthly_forecast,
    daily_day_weight_profile
):
    forecast_month = pd.Timestamp(forecast_month)

    if forecast_month.month in [1, 5, 9]:
        forecast_month_type = "Cohort Close Month"
    else:
        forecast_month_type = "Normal Month"

    forecast_dates = pd.date_range(
        start=forecast_month,
        end=forecast_month + pd.offsets.MonthEnd(0),
        freq="D"
    )

    daily_forecast_df = pd.DataFrame({
        "date": forecast_dates
    })

    daily_forecast_df["day"] = daily_forecast_df["date"].dt.day

    month_weights = daily_day_weight_profile[
        daily_day_weight_profile["month_type"] == forecast_month_type
    ][["day", "normalized_weight"]]

    daily_forecast_df = daily_forecast_df.merge(
        month_weights,
        on="day",
        how="left"
    )

    daily_forecast_df["normalized_weight"] = (
        daily_forecast_df["normalized_weight"].fillna(0)
    )

    daily_forecast_df["normalized_weight"] = (
        daily_forecast_df["normalized_weight"]
        / daily_forecast_df["normalized_weight"].sum()
    )

    daily_forecast_df["forecasted_enrollments"] = (
        monthly_forecast * daily_forecast_df["normalized_weight"]
    )

    daily_forecast_df["forecasted_enrollments"] = (
        daily_forecast_df["forecasted_enrollments"]
        .round()
        .astype(int)
    )

    difference = (
        monthly_forecast
        - daily_forecast_df["forecasted_enrollments"].sum()
    )

    daily_forecast_df.loc[
        daily_forecast_df.index[-1],
        "forecasted_enrollments"
    ] += difference

    daily_forecast_df["month"] = forecast_month
    daily_forecast_df["month_type"] = forecast_month_type

    daily_forecast_df["week_start"] = (
        daily_forecast_df["date"]
        - pd.to_timedelta(daily_forecast_df["date"].dt.weekday, unit="D")
    )

    return daily_forecast_df


# ---------------------------------------
# Daily pace tracking
# ---------------------------------------

def get_daily_pace():
    monthly_forecast_df = forecast_next_6_months()

    current_month = pd.Timestamp(
        monthly_forecast_df["month"].iloc[0]
    )

    monthly_forecast = int(
        monthly_forecast_df["forecasted_enrollments"].iloc[0]
    )

    daily_day_weight_profile = build_daily_day_weight_profile()

    forecast_daily = create_daily_forecast_allocation(
        forecast_month=current_month,
        monthly_forecast=monthly_forecast,
        daily_day_weight_profile=daily_day_weight_profile
    )

    actual_daily = load_daily_enrollment_data()

    actual_current_month = actual_daily[
        actual_daily["date"].dt.to_period("M")
        == current_month.to_period("M")
    ].copy()

    pace_df = forecast_daily.merge(
        actual_current_month[["date", "enrollments"]],
        on="date",
        how="left"
    )

    pace_df = pace_df.rename(columns={
        "enrollments": "actual_enrollments"
    })

    pace_df["actual_enrollments"] = (
        pace_df["actual_enrollments"].fillna(0).astype(int)
    )

    pace_df["cumulative_forecast"] = (
        pace_df["forecasted_enrollments"].cumsum()
    )

    pace_df["cumulative_actual"] = (
        pace_df["actual_enrollments"].cumsum()
    )

    latest_actual_date = actual_current_month["date"].max()

    if pd.isna(latest_actual_date):
        latest_actual_date = current_month - pd.Timedelta(days=1)

    today_df = pace_df[
        pace_df["date"] <= latest_actual_date
    ]

    actual_to_date = int(today_df["actual_enrollments"].sum())
    expected_to_date = int(today_df["forecasted_enrollments"].sum())

    pace_variance = actual_to_date - expected_to_date

    if expected_to_date > 0:
        pace_achievement_pct = round(
            (actual_to_date / expected_to_date) * 100,
            2
        )
    else:
        pace_achievement_pct = 0

    remaining_target = monthly_forecast - actual_to_date

    days_remaining = int(
        (forecast_daily["date"].max() - latest_actual_date).days
    )

    if days_remaining > 0:
        required_daily_pace = round(
            remaining_target / days_remaining,
            2
        )
    else:
        required_daily_pace = 0

    weekly_forecast = (
        forecast_daily
        .groupby("week_start")["forecasted_enrollments"]
        .sum()
        .reset_index()
    )

    pace_df["date"] = pace_df["date"].astype(str)
    pace_df["month"] = pace_df["month"].astype(str)
    pace_df["week_start"] = pace_df["week_start"].astype(str)

    weekly_forecast["week_start"] = (
        weekly_forecast["week_start"].astype(str)
    )

    return {
        "month": str(current_month.date()),
        "monthly_forecast": monthly_forecast,
        "latest_actual_date": str(latest_actual_date.date()),
        "actual_to_date": actual_to_date,
        "expected_to_date": expected_to_date,
        "pace_variance": int(pace_variance),
        "pace_achievement_pct": pace_achievement_pct,
        "remaining_target": int(remaining_target),
        "days_remaining": days_remaining,
        "required_daily_pace": required_daily_pace,
        "daily_data": to_records(pace_df),
        "weekly_forecast": to_records(weekly_forecast)
    }

# ---------------------------------------
# Save daily forecast history
# ---------------------------------------
    
def save_daily_forecast_history():
    pace_data = get_daily_pace()

    snapshot_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    rows = []

    for row in pace_data["daily_data"]:
        rows.append({
            "snapshot_date": snapshot_date,
            "forecast_month": pace_data["month"],
            "forecast_date": row["date"],
            "forecasted_enrollments": row["forecasted_enrollments"],
            "month_type": row["month_type"],
            "monthly_forecast": pace_data["monthly_forecast"],
            "model_version": MODEL_VERSION
        })

    payload = {
        "rows": rows
    }

    response = requests.post(
        FORECAST_HISTORY_WEB_APP_URL,
        json=payload,
        timeout=30
    )

    return {
        "status_code": response.status_code,
        "response": response.text,
        "rows_sent": len(rows),
        "forecast_month": pace_data["month"],
        "snapshot_date": snapshot_date
    }


# ---------------------------------------
# Local test
# ---------------------------------------

if __name__ == "__main__":
    print("Rolling 6-month forecast:")
    print(forecast_next_6_months())

    print("\nCurrent-year projection:")
    print(get_current_year_projection())

    print("\nDaily pace:")
    print(get_daily_pace())
