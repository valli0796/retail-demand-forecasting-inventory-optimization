import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = "data/processed/sales_data_cleaned.csv"
OUTPUT_PATH = "data/processed/product_forecast_results.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["store_id", "product_id", "date"]
    ).reset_index(drop=True)

    return df


def create_product_daily_data(df):

    daily = (
        df.groupby(
            ["store_id", "product_id", "date"],
            as_index=False
        )
        .agg({
            "demand": "sum",
            "inventory_level": "mean",
            "price": "mean",
            "discount": "mean",
            "competitor_pricing": "mean",
            "promotion": "mean",
            "epidemic": "mean"
        })
    )

    daily = daily.sort_values(
        ["store_id", "product_id", "date"]
    ).reset_index(drop=True)

    group = daily.groupby(
        ["store_id", "product_id"]
    )

    # Lag features
    daily["lag_1"] = group["demand"].shift(1)
    daily["lag_7"] = group["demand"].shift(7)
    daily["lag_14"] = group["demand"].shift(14)
    daily["lag_28"] = group["demand"].shift(28)

    # Rolling features
    daily["rolling_mean_7"] = (
        group["demand"]
        .shift(1)
        .groupby(
            [daily["store_id"], daily["product_id"]]
        )
        .transform(
            lambda x: x.rolling(7).mean()
        )
    )

    daily["rolling_mean_14"] = (
        group["demand"]
        .shift(1)
        .groupby(
            [daily["store_id"], daily["product_id"]]
        )
        .transform(
            lambda x: x.rolling(14).mean()
        )
    )

    daily["rolling_mean_28"] = (
        group["demand"]
        .shift(1)
        .groupby(
            [daily["store_id"], daily["product_id"]]
        )
        .transform(
            lambda x: x.rolling(28).mean()
        )
    )

    daily["rolling_std_7"] = (
        group["demand"]
        .shift(1)
        .groupby(
            [daily["store_id"], daily["product_id"]]
        )
        .transform(
            lambda x: x.rolling(7).std()
        )
    )

    # Calendar features
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

    # Remove rows where lag/rolling features are unavailable
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


def train_model(train, test):

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

    print("\nTraining product-level XGBoost...")

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

    predictions = model.predict(
        X_test
    )

    return model, predictions, features


def main():

    print("Loading data...")

    df = load_data()

    print(
        "Original records:",
        len(df)
    )

    combinations = (
        df[
            ["store_id", "product_id"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print(
        "Store-product combinations:",
        combinations
    )

    daily = create_product_daily_data(df)

    print(
        "Product-level daily records:",
        len(daily)
    )

    # Chronological 80/20 split
    split_date = daily["date"].quantile(0.8)

    train = daily[
        daily["date"] <= split_date
    ].copy()

    test = daily[
        daily["date"] > split_date
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

    model, predictions, features = train_model(
        train,
        test
    )

    mae, rmse, mape = calculate_metrics(
        test["demand"].values,
        predictions
    )

    print("\n==============================")
    print("PRODUCT-LEVEL XGBOOST RESULTS")
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

    results = test[
        [
            "store_id",
            "product_id",
            "date",
            "demand",
            "inventory_level"
        ]
    ].copy()

    results["Forecast"] = predictions

    results["Forecast"] = (
        results["Forecast"]
        .clip(lower=0)
    )

    results["Forecast Error"] = (
        results["demand"]
        - results["Forecast"]
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nSample predictions:")

    print(
        results.head(15)
        .to_string(index=False)
    )

    print("\nForecast results saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()