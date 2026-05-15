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
    """
    Converts either Excel serial dates or date strings into pandas datetime.
    """

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
        # January cohort: Oct → Jan
        10: 1,
        11: 2,
        12: 3,
        1: 4,

        # May cohort: Feb → May
        2: 1,
        3: 2,
        4: 3,
        5: 4,

        # September cohort: Jun → Sep
        6: 1,
        7: 2,
        8: 3,
        9: 4,
    }

    return stage_mapping[month]


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

    # Standardize column names
    enroll_df.columns = enroll_df.columns.str.strip().str.lower()
    apps_df.columns = apps_df.columns.str.strip().str.lower()

    # Rename month columns
    enroll_df = enroll_df.rename(columns={
        "enrolment_month": "month",
        "enrollment_month": "month",
    })

    apps_df = apps_df.rename(columns={
        "app_month": "month",
    })

    # Convert month columns
    enroll_df["month"] = enroll_df["month"].apply(convert_month_column)
    apps_df["month"] = apps_df["month"].apply(convert_month_column)

    # Keep required columns
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

    # Merge datasets
    df = pd.merge(
        enroll_df,
        apps_df,
        on="month",
        how="inner"
    )

    # Standard spelling
    df = df.rename(columns={
        "enrolments": "enrollments"
    })

    # Sort chronologically
    df = df.sort_values("month").reset_index(drop=True)

    # Remove launch month distortion
    df = df[df["month"] != pd.Timestamp("2023-05-01")]
    df = df.reset_index(drop=True)

    return df


# ---------------------------------------
# Forecast function
# ---------------------------------------

def forecast_next_months(periods=12):
    """
    Generate recursive enrollment forecasts for the next N months.

    Default:
    - 12 months

    The function:
    - loads latest live data
    - creates model features for each future month
    - predicts recursively
    - returns forecast dataframe
    """

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

        # -------------------------------
        # Seasonal features
        # -------------------------------

        row["month_sin"] = np.sin(
            2 * np.pi * month_number / 12
        )

        row["month_cos"] = np.cos(
            2 * np.pi * month_number / 12
        )

        # -------------------------------
        # Cohort features
        # -------------------------------

        row["is_cohort_close_month"] = int(
            month_number in [1, 5, 9]
        )

        row["cohort_stage"] = create_cohort_stage(
            month_number
        )

        # -------------------------------
        # Lagged enrollment features
        # -------------------------------

        row["enrollments_lag_1"] = (
            forecast_df["enrollments"].iloc[-1]
        )

        row["enrollments_lag_2"] = (
            forecast_df["enrollments"].iloc[-2]
        )

        row["enrollments_lag_12"] = (
            forecast_df["enrollments"].iloc[-12]
        )

        # -------------------------------
        # Lagged application features
        # -------------------------------

        row["apps_accepted_lag_1"] = (
            forecast_df["applications_accepted"].iloc[-1]
        )

        row["apps_accepted_lag_2"] = (
            forecast_df["applications_accepted"].iloc[-2]
        )

        row["apps_accepted_roll_3"] = (
            forecast_df["applications_accepted"]
            .iloc[-3:]
            .mean()
        )

        # -------------------------------
        # Optional features
        # -------------------------------

        if "days_in_month" in features:
            row["days_in_month"] = future_month.days_in_month

        if "programs" in features:
            row["programs"] = forecast_df["programs"].iloc[-1]

        if "enrollments_roll_3" in features:
            row["enrollments_roll_3"] = (
                forecast_df["enrollments"]
                .iloc[-3:]
                .mean()
            )

        if "conversion_rate_lag_1" in features:
            row["conversion_rate_lag_1"] = (
                forecast_df["enrollments"].iloc[-1]
                / forecast_df["applications_accepted"].iloc[-1]
            )

        # -------------------------------
        # Predict
        # -------------------------------

        X_future = pd.DataFrame([row])[features]

        prediction = ridge_model.predict(X_future)[0]
        prediction = max(0, prediction)

        # Enrollment is a count, so return whole number
        prediction = int(np.floor(prediction))

        predictions.append(prediction)

        # -------------------------------
        # Recursive update
        # -------------------------------

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


# ---------------------------------------
# Convenience wrappers
# ---------------------------------------

def forecast_next_6_months():
    return forecast_next_months(periods=6)


def forecast_next_12_months():
    return forecast_next_months(periods=12)


# ---------------------------------------
# Local test
# ---------------------------------------

if __name__ == "__main__":
    forecast = forecast_next_12_months()
    print(forecast)
