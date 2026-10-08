import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


DATA_PATH = "data/processed/sales_data_cleaned.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    return df


def create_daily_features(df):

    daily = (
        df.groupby("date")
        .agg({
            "demand": "sum",
            "inventory_level": "mean",
            "price": "mean",
            "discount": "mean",
            "competitor_pricing": "mean",
            "promotion": "mean",
            "epidemic": "mean"
        })
        .reset_index()
    )

    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["day"] = daily["date"].dt.day
    daily["week_of_year"] = (
        daily["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    daily["is_weekend"] = (
        daily["day_of_week"] >= 5
    ).astype(int)

    # Historical demand features
    daily["lag_1"] = daily["demand"].shift(1)
    daily["lag_7"] = daily["demand"].shift(7)
    daily["lag_14"] = daily["demand"].shift(14)
    daily["lag_28"] = daily["demand"].shift(28)

    # Rolling statistics use only previous observations
    daily["rolling_mean_7"] = (
        daily["demand"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    daily["rolling_mean_14"] = (
        daily["demand"]
        .shift(1)
        .rolling(14)
        .mean()
    )

    daily["rolling_mean_28"] = (
        daily["demand"]
        .shift(1)
        .rolling(28)
        .mean()
    )

    daily["rolling_std_7"] = (
        daily["demand"]
        .shift(1)
        .rolling(7)
        .std()
    )

    daily = daily.dropna().reset_index(drop=True)

    return daily


def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    non_zero = actual != 0

    mape = np.mean(
        np.abs(
            (
                actual[non_zero]
                - predicted[non_zero]
            )
            / actual[non_zero]
        )
    ) * 100

    return mae, rmse, mape


def train_xgboost(train, test):

    features = [
        "inventory_level",
        "price",
        "discount",
        "competitor_pricing",
        "promotion",
        "epidemic",
        "day_of_week",
        "month",
        "day",
        "week_of_year",
        "is_weekend",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_28",
        "rolling_std_7"
    ]

    X_train = train[features]
    y_train = train["demand"]

    X_test = test[features]

    print("\nTraining leakage-free XGBoost...")

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    return model, predictions, features


def main():

    print("Loading data...")

    df = load_data()

    print(
        "Original records:",
        len(df)
    )

    daily = create_daily_features(df)

    print(
        "Daily records:",
        len(daily)
    )

    # Chronological split
    split_index = int(
        len(daily) * 0.8
    )

    train = daily.iloc[
        :split_index
    ].copy()

    test = daily.iloc[
        split_index:
    ].copy()

    print(
        "\nTraining records:",
        len(train)
    )

    print(
        "Testing records:",
        len(test)
    )

    print(
        "\nTraining period:",
        train["date"].min(),
        "to",
        train["date"].max()
    )

    print(
        "Testing period:",
        test["date"].min(),
        "to",
        test["date"].max()
    )

    model, predictions, features = train_xgboost(
        train,
        test
    )

    mae, rmse, mape = calculate_metrics(
        test["demand"],
        predictions
    )

    print("\n==============================")
    print("XGBOOST RESULTS")
    print("==============================")

    print(
        "MAE:",
        round(mae, 2)
    )

    print(
        "RMSE:",
        round(rmse, 2)
    )

    print(
        "MAPE:",
        round(mape, 2),
        "%"
    )

    comparison = pd.DataFrame({
        "Date": test["date"],
        "Actual": test["demand"],
        "Predicted": predictions
    })

    print("\nActual vs Predicted:")

    print(
        comparison.head(10).to_string(
            index=False
        )
    )

    # Feature importance
    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    print("\n==============================")
    print("FEATURE IMPORTANCE")
    print("==============================")

    print(
        importance.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()