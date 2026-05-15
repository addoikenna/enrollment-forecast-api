import pandas as pd
import numpy as np
import joblib


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

ENROLLMENT_GID = "1781196956"
APPLICATIONS_GID = "3304352"


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


# ---------------------------------------
# Load live Google Sheet data
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

    df = df[df["month"] != pd.Timestamp("2023-05-01")]
    df = df.reset_index(drop=True)

    return df


# ---------------------------------------
# Forecast function
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
    """
    Current year logic:
    - Actuals: Jan to previous month
    - Forecast: current month to Dec
    """

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
# Local test
# ---------------------------------------

if __name__ == "__main__":
    print("Rolling 6-month forecast:")
    print(forecast_next_6_months())

    print("\nCurrent-year projection:")
    print(get_current_year_projection())
