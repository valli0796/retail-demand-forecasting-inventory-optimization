import pandas as pd
import numpy as np

FORECAST_PATH = "data/processed/product_forecast_results.csv"
OUTPUT_PATH = "data/processed/forecast_inventory_recommendations.csv"

LEAD_TIME_DAYS = 7
SERVICE_LEVEL = 0.95
Z_SCORE = 1.645

ORDERING_COST = 100
HOLDING_COST = 5


def load_forecast_data():

    df = pd.read_csv(
        FORECAST_PATH
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df


def calculate_inventory_metrics(group):

    forecast_demand = group["Forecast"]

    current_inventory = group[
        "inventory_level"
    ].iloc[-1]

    average_forecast = forecast_demand.mean()

    demand_std = forecast_demand.std()

    if pd.isna(demand_std):
        demand_std = 0

    # Safety Stock
    safety_stock = (
        Z_SCORE
        * demand_std
        * np.sqrt(LEAD_TIME_DAYS)
    )

    # Reorder Point
    reorder_point = (
        average_forecast
        * LEAD_TIME_DAYS
        + safety_stock
    )

    # Annual demand
    annual_demand = (
        average_forecast
        * 365
    )

    # EOQ
    if annual_demand > 0:

        eoq = np.sqrt(
            (
                2
                * annual_demand
                * ORDERING_COST
            )
            / HOLDING_COST
        )

    else:

        eoq = 0

    # Recommended order quantity
    if current_inventory <= reorder_point:

        recommended_order_qty = max(
            eoq,
            reorder_point - current_inventory
        )

        alert = "REORDER"

    elif current_inventory <= (
        reorder_point * 1.25
    ):

        recommended_order_qty = 0

        alert = "LOW STOCK"

    else:

        recommended_order_qty = 0

        alert = "NORMAL"

    return pd.Series({

        "Average Forecast Demand":
            average_forecast,

        "Demand Std":
            demand_std,

        "Current Inventory":
            current_inventory,

        "Safety Stock":
            safety_stock,

        "Reorder Point":
            reorder_point,

        "Annual Demand":
            annual_demand,

        "EOQ":
            eoq,

        "Recommended Order Qty":
            recommended_order_qty,

        "Alert":
            alert
    })


def main():

    print("Loading product-level forecasts...")

    df = load_forecast_data()

    print(
        "Forecast records:",
        len(df)
    )

    print(
        "Store-product combinations:",
        df[
            ["store_id", "product_id"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print("\nCalculating inventory recommendations...")

    recommendations = (
        df.groupby(
            ["store_id", "product_id"]
        )
        .apply(
            calculate_inventory_metrics,
            include_groups=False
        )
        .reset_index()
    )

    # Round numerical columns
    numerical_columns = [
        "Average Forecast Demand",
        "Demand Std",
        "Current Inventory",
        "Safety Stock",
        "Reorder Point",
        "Annual Demand",
        "EOQ",
        "Recommended Order Qty"
    ]

    recommendations[
        numerical_columns
    ] = recommendations[
        numerical_columns
    ].round(2)

    print("\n==============================")
    print("INVENTORY OPTIMIZATION RESULTS")
    print("==============================")

    print(
        "\nTotal recommendations:",
        len(recommendations)
    )

    print("\nAlert Summary:")

    print(
        recommendations[
            "Alert"
        ]
        .value_counts()
    )

    print("\nTop reorder recommendations:")

    reorder = recommendations[
        recommendations["Alert"] == "REORDER"
    ].sort_values(
        "Recommended Order Qty",
        ascending=False
    )

    print(
        reorder.head(15)
        .to_string(index=False)
    )

    recommendations.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "\nRecommendations saved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()