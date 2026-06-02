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
# Google Sheet IDs, GIDs, and URLs
# ---------------------------------------

ENROLLMENT_SHEET_ID = "1bkUlCdL0VpCy-2Gm17MfTlC2lbv-qQoNQbasN2cGniw"
APPLICATIONS_SHEET_ID = "1ry6T6I0qHbme3Bb1yVTSZyop9Cn1ictFU93V-lOlrfo"
FORECAST_HISTORY_SHEET_ID = "1Fd2rBfeyLwzweNOFLMXjOTqhDzqdm1wmDdP2bD4cJo4"

ENROLLMENT_GID = "1781196956"
APPLICATIONS_GID = "3304352"
DAILY_ENROLLMENT_GID = "604024081"
FORECAST_HISTORY_GID = "0"

FORECAST_HISTORY_WEB_APP_URL = (
    "https://script.google.com/macros/s/"
    "AKfycbyueag5B7rLm6zxy2g88olVH8R_ftGI9ZEAKBJ_mmnJFLRSXY5nnQbYm229QDLmgKT4jg/exec"
)

MODEL_VERSION = "Ridge_v1"


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
        "program",
        "enrollments"
    ]]

    daily_df["program"] = daily_df["program"].astype(str).str.strip()

    daily_df = daily_df.sort_values("date").reset_index(drop=True)

    return daily_df


# ---------------------------------------
# Get eligible programs
# ---------------------------------------

def get_eligible_programs(min_history_months=3):
    daily_df = load_daily_enrollment_data()

    daily_df["month_period"] = (
        daily_df["date"].dt.to_period("M").astype(str)
    )

    program_history = (
        daily_df
        .groupby("program")["month_period"]
        .nunique()
        .reset_index()
        .rename(columns={"month_period": "months_active"})
    )

    eligible = program_history[
        program_history["months_active"] >= min_history_months
    ].copy()

    eligible = eligible.sort_values("program")

    programs = eligible["program"].tolist()

    return {
        "programs": ["All Programs"] + programs
    }


#----------------------------------------
# Get program share
#----------------------------------------

def get_program_shares(months_back=6, min_history_months=3):
    daily_df = load_daily_enrollment_data()

    daily_df["month_period"] = daily_df["date"].dt.to_period("M")

    max_date = daily_df["date"].max()
    cutoff_date = max_date - pd.DateOffset(months=months_back)

    recent_df = daily_df[
        daily_df["date"] >= cutoff_date
    ].copy()

    program_history = (
        daily_df
        .groupby("program")["month_period"]
        .nunique()
        .reset_index()
        .rename(columns={"month_period": "months_active"})
    )

    eligible_programs = program_history[
        program_history["months_active"] >= min_history_months
    ]["program"]

    recent_df = recent_df[
        recent_df["program"].isin(eligible_programs)
    ]

    program_totals = (
        recent_df
        .groupby("program")["enrollments"]
        .sum()
        .reset_index()
    )

    total_enrollments = program_totals["enrollments"].sum()

    program_totals["share"] = (
        program_totals["enrollments"] / total_enrollments
    )

    program_totals = program_totals.sort_values(
        "share",
        ascending=False
    )

    return program_totals


# ---------------------------------------
# Load forecast history
# ---------------------------------------

def load_forecast_history():
    history_url = google_sheet_csv_url(
        FORECAST_HISTORY_SHEET_ID,
        FORECAST_HISTORY_GID
    )

    history_df = pd.read_csv(history_url)

    history_df.columns = history_df.columns.str.strip().str.lower()

    history_df["snapshot_date"] = pd.to_datetime(history_df["snapshot_date"])
    history_df["forecast_month"] = pd.to_datetime(history_df["forecast_month"])
    history_df["forecast_date"] = pd.to_datetime(history_df["forecast_date"])

    history_df["forecasted_enrollments"] = (
        history_df["forecasted_enrollments"].astype(int)
    )

    history_df["monthly_forecast"] = (
        history_df["monthly_forecast"].astype(int)
    )

    # Keep only the latest snapshot for each forecast month and forecast date
    history_df = history_df.sort_values("snapshot_date")
    
    history_df = history_df.drop_duplicates(
        subset=["forecast_month", "forecast_date"],
        keep="last"
    )
    
    history_df = history_df.reset_index(drop=True)

    return history_df


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
# Forecast history months
# ---------------------------------------

def get_forecast_history_months():
    history_df = load_forecast_history()

    history_df["month_key"] = (
        history_df["forecast_month"]
        .dt.to_period("M")
        .astype(str)
    )

    months = (
        history_df["month_key"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    return {
        "months": months
    }


# ---------------------------------------
# Daily pace tracking
# ---------------------------------------

def get_daily_pace(month=None):

    if month is None:
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

        forecast_daily = forecast_daily.sort_values("date").reset_index(drop=True)

    else:
        history_df = load_forecast_history()

        current_month = pd.Timestamp(month)

        forecast_daily = history_df[
            history_df["forecast_month"].dt.to_period("M")
            == current_month.to_period("M")
        ].copy()

        if len(forecast_daily) == 0:
            raise ValueError(f"No forecast history found for {month}")

        monthly_forecast = int(
            forecast_daily["monthly_forecast"].iloc[0]
        )

        forecast_daily = forecast_daily.rename(columns={
            "forecast_date": "date"
        })

        forecast_daily["month"] = current_month

        forecast_daily["week_start"] = (
            forecast_daily["date"]
            - pd.to_timedelta(forecast_daily["date"].dt.weekday, unit="D")
        )

        forecast_daily = forecast_daily.sort_values("date").reset_index(drop=True)

    actual_daily = load_daily_enrollment_data()

    actual_current_month = actual_daily[
        actual_daily["date"].dt.to_period("M")
        == current_month.to_period("M")
    ].copy()

    actual_current_month = (
        actual_current_month
        .sort_values("date")
        .reset_index(drop=True)
    )

    pace_df = forecast_daily.merge(
        actual_current_month[["date", "enrollments"]],
        on="date",
        how="left"
    )

    pace_df = pace_df.rename(columns={
        "enrollments": "actual_enrollments"
    })

    pace_df = pace_df.sort_values("date").reset_index(drop=True)

    pace_df["actual_enrollments"] = (
        pace_df["actual_enrollments"]
        .fillna(0)
        .astype(int)
    )

    pace_df["cumulative_forecast"] = (
        pace_df["forecasted_enrollments"]
        .cumsum()
    )

    pace_df["cumulative_actual"] = (
        pace_df["actual_enrollments"]
        .cumsum()
    )

    pace_df["pace_variance"] = (
        pace_df["cumulative_actual"]
        - pace_df["cumulative_forecast"]
    )

    # ---------------------------------------
    # Determine actual and forecast dates
    # ---------------------------------------

    latest_actual_date = actual_current_month["date"].max()

    if pd.isna(latest_actual_date):
        latest_actual_date = current_month - pd.Timedelta(days=1)

    today = pd.Timestamp.today().normalize()

    month_start = current_month
    month_end = forecast_daily["date"].max()

    if today < month_start:
        pace_as_of_date = month_start
    elif today > month_end:
        pace_as_of_date = month_end
    else:
        pace_as_of_date = today

    # ---------------------------------------
    # KPI calculations
    # ---------------------------------------

    actual_to_date = int(
        pace_df[
            pace_df["date"] <= pace_as_of_date
        ]["actual_enrollments"].sum()
    )

    expected_to_date = int(
        pace_df[
            pace_df["date"] <= pace_as_of_date
        ]["forecasted_enrollments"].sum()
    )

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
        (month_end - pace_as_of_date).days
    )

    if days_remaining > 0:
        required_daily_pace = round(
            remaining_target / days_remaining,
            2
        )
    else:
        required_daily_pace = 0

    # ---------------------------------------
    # Today / Yesterday metrics
    # ---------------------------------------

    today_row = pace_df[
        pace_df["date"] == pace_as_of_date
    ]

    if len(today_row) > 0:
        enrollment_today = int(
            today_row["actual_enrollments"].iloc[0]
        )

        forecasted_enrollment_today = int(
            today_row["forecasted_enrollments"].iloc[0]
        )
    else:
        enrollment_today = 0
        forecasted_enrollment_today = 0

    yesterday = pace_as_of_date - pd.Timedelta(days=1)

    # Pull yesterday from full actual data, not only selected month
    yesterday_actual = actual_daily[
        actual_daily["date"] == yesterday
    ]

    if len(yesterday_actual) > 0:
        enrollment_yesterday = int(
            yesterday_actual["enrollments"].iloc[0]
        )
    else:
        enrollment_yesterday = 0

    if enrollment_yesterday > 0:
        enrollment_today_change_pct = round(
            ((enrollment_today - enrollment_yesterday) / enrollment_yesterday) * 100,
            2
        )
    else:
        enrollment_today_change_pct = None

    weekly_forecast = (
        forecast_daily
        .groupby("week_start")["forecasted_enrollments"]
        .sum()
        .reset_index()
        .sort_values("week_start")
        .reset_index(drop=True)
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
        "pace_as_of_date": str(pace_as_of_date.date()),
        "actual_to_date": actual_to_date,
        "expected_to_date": expected_to_date,
        "pace_variance": int(pace_variance),
        "pace_achievement_pct": pace_achievement_pct,
        "remaining_target": int(remaining_target),
        "days_remaining": days_remaining,
        "required_daily_pace": required_daily_pace,
        "enrollment_today": enrollment_today,
        "enrollment_yesterday": enrollment_yesterday,
        "enrollment_today_change_pct": enrollment_today_change_pct,
        "forecasted_enrollment_today": forecasted_enrollment_today,
        "daily_data": to_records(pace_df),
        "weekly_forecast": to_records(weekly_forecast)
    }

# ---------------------------------------
# Save daily forecast history
# ---------------------------------------

def save_daily_forecast_history():
    pace_data = get_daily_pace()

    snapshot_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    forecast_month = pace_data["month"]

    # ---------------------------------------
    # Check if forecast month already exists
    # ---------------------------------------

    try:
        history_df = load_forecast_history()

        existing_months = (
            history_df["forecast_month"]
            .dt.strftime("%Y-%m-%d")
            .drop_duplicates()
            .tolist()
        )

        if forecast_month in existing_months:
            return {
                "status": "skipped",
                "message": f"Forecast already exists for {forecast_month}",
                "forecast_month": forecast_month,
                "snapshot_date": snapshot_date,
                "rows_sent": 0
            }

    except Exception:
        # If history sheet is empty or unreadable, continue and save forecast
        pass

    # ---------------------------------------
    # Prepare rows for saving
    # ---------------------------------------

    rows = []

    for row in pace_data["daily_data"]:
        rows.append({
            "snapshot_date": snapshot_date,
            "forecast_month": forecast_month,
            "forecast_date": row["date"],
            "forecasted_enrollments": row["forecasted_enrollments"],
            "month_type": row["month_type"],
            "monthly_forecast": pace_data["monthly_forecast"],
            "model_version": MODEL_VERSION
        })

    payload = {
        "rows": rows
    }

    # ---------------------------------------
    # Send rows to Google Apps Script
    # ---------------------------------------

    response = requests.post(
        FORECAST_HISTORY_WEB_APP_URL,
        json=payload,
        timeout=30
    )

    return {
        "status": "saved",
        "status_code": response.status_code,
        "response": response.text,
        "rows_sent": len(rows),
        "forecast_month": forecast_month,
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

    print("\nForecast history months:")
    print(get_forecast_history_months())

    print("\nDaily pace:")
    print(get_daily_pace())
