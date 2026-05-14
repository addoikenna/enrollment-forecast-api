import pandas as pd
import numpy as np
import joblib


# ---------------------------------------
# Load model assets
# ---------------------------------------

ridge_model = joblib.load(
    "models/ridge_enrollment_model.pkl"
)

features = joblib.load(
    "models/ridge_model_features.pkl"
)


# ---------------------------------------
# Load dataset
# ---------------------------------------

df = pd.read_csv(
    "data/model_dataset.csv"
)

df["month"] = pd.to_datetime(df["month"])

df = df.sort_values("month").reset_index(drop=True)


# ---------------------------------------
# Cohort stage helper
# ---------------------------------------

def create_cohort_stage(month):

    stage_mapping = {

        10: 1,
        11: 2,
        12: 3,
        1: 4,

        2: 1,
        3: 2,
        4: 3,
        5: 4,

        6: 1,
        7: 2,
        8: 3,
        9: 4
    }

    return stage_mapping[month]


# ---------------------------------------
# Forecast function
# ---------------------------------------

def forecast_next_6_months():

    forecast_df = df.copy()

    predictions = []

    future_months = pd.date_range(
        start=forecast_df["month"].max()
        + pd.offsets.MonthBegin(1),

        periods=6,
        freq="MS"
    )

    for future_month in future_months:

        month_number = future_month.month

        row = {}

        # -----------------------------------
        # Seasonal features
        # -----------------------------------

        row["month_sin"] = np.sin(
            2 * np.pi * month_number / 12
        )

        row["month_cos"] = np.cos(
            2 * np.pi * month_number / 12
        )

        # -----------------------------------
        # Cohort features
        # -----------------------------------

        row["is_cohort_close_month"] = int(
            month_number in [1, 5, 9]
        )

        row["cohort_stage"] = (
            create_cohort_stage(month_number)
        )

        # -----------------------------------
        # Lag features
        # -----------------------------------

        row["enrollments_lag_1"] = (
            forecast_df["enrollments"].iloc[-1]
        )

        row["enrollments_lag_2"] = (
            forecast_df["enrollments"].iloc[-2]
        )

        row["enrollments_lag_12"] = (
            forecast_df["enrollments"].iloc[-12]
        )

        # -----------------------------------
        # Applications features
        # -----------------------------------

        row["apps_accepted_lag_1"] = (
            forecast_df["applications_accepted"]
            .iloc[-1]
        )

        row["apps_accepted_lag_2"] = (
            forecast_df["applications_accepted"]
            .iloc[-2]
        )

        row["apps_accepted_roll_3"] = (
            forecast_df["applications_accepted"]
            .iloc[-3:]
            .mean()
        )

        # -----------------------------------
        # Optional features
        # -----------------------------------

        if "days_in_month" in features:

            row["days_in_month"] = (
                future_month.days_in_month
            )

        if "programs" in features:

            row["programs"] = (
                forecast_df["programs"].iloc[-1]
            )

        if "enrollments_roll_3" in features:

            row["enrollments_roll_3"] = (
                forecast_df["enrollments"]
                .iloc[-3:]
                .mean()
            )

        # -----------------------------------
        # Predict
        # -----------------------------------

        X_future = pd.DataFrame([row])[features]

        prediction = ridge_model.predict(
            X_future
        )[0]

        prediction = max(0, prediction)

        predictions.append(prediction)

        # -----------------------------------
        # Recursive update
        # -----------------------------------

        new_row = forecast_df.iloc[-1].copy()

        new_row["month"] = future_month
        new_row["enrollments"] = prediction

        forecast_df = pd.concat(
            [
                forecast_df,
                pd.DataFrame([new_row])
            ],
            ignore_index=True
        )

    # ---------------------------------------
    # Final forecast dataframe
    # ---------------------------------------

    forecast_output = pd.DataFrame({

        "month": future_months,

        "forecasted_enrollments":
            predictions
    })

    return forecast_output


# ---------------------------------------
# Run locally
# ---------------------------------------

if __name__ == "__main__":

    forecast = forecast_next_6_months()

    print(forecast)
