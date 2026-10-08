import pandas as pd
import numpy as np

INPUT_PATH = "data/processed/retail_sales_cleaned.csv"
OUTPUT_PATH = "data/processed/inventory_recommendations.csv"

LEAD_TIME_DAYS = 7
SERVICE_LEVEL = 0.95
Z_SCORE = 1.645

ORDERING_COST = 100
HOLDING_COST = 5


def calculate_safety_stock(
    demand_std,
    lead_time_days=LEAD_TIME_DAYS,
    z_score=Z_SCORE
):
    return (
        z_score
        * demand_std
        * np.sqrt(lead_time_days)
    )


def calculate_reorder_point(
    average_demand,
    safety_stock,
    lead_time_days=LEAD_TIME_DAYS
):
    return (
        average_demand
        * lead_time_days
        + safety_stock
    )


def calculate_eoq(
    annual_demand,
    ordering_cost=ORDERING_COST,
    holding_cost=HOLDING_COST
):
    if annual_demand <= 0:
        return 0

    return np.sqrt(
        (
            2
            * annual_demand
            * ordering_cost
        )
        / holding_cost
    )


def calculate_inventory_metrics(group):
    average_demand = group["demand"].mean()
    demand_std = group["demand"].std()

    if pd.isna(demand_std):
        demand_std = 0

    current_inventory = group["inventory_level"].iloc[-1]

    safety_stock = calculate_safety_stock(
        demand_std
    )

    reorder_point = calculate_reorder_point(
        average_demand,
        safety_stock
    )

    annual_demand = average_demand * 365

    eoq = calculate_eoq(
        annual_demand
    )

    if current_inventory <= reorder_point:
        recommended_order_qty = max(
            eoq,
            reorder_point - current_inventory
        )
        alert = "REORDER"

    elif current_inventory <= reorder_point * 1.25:
        recommended_order_qty = 0
        alert = "LOW STOCK"

    else:
        recommended_order_qty = 0
        alert = "NORMAL"

    return pd.Series({
        "Average Daily Demand": average_demand,
        "Demand Std": demand_std,
        "Current Inventory": current_inventory,
        "Safety Stock": safety_stock,
        "Reorder Point": reorder_point,
        "Annual Demand": annual_demand,
        "EOQ": eoq,
        "Recommended Order Qty": recommended_order_qty,
        "Alert": alert
    })


def generate_inventory_recommendations():
    print("Loading cleaned retail data...")

    df = pd.read_csv(INPUT_PATH)

    df["date"] = pd.to_datetime(df["date"])

    print("Records:", len(df))

    recommendations = (
        df.groupby(["store_id", "product_id"])
        .apply(
            calculate_inventory_metrics,
            include_groups=False
        )
        .reset_index()
    )

    numerical_columns = [
        "Average Daily Demand",
        "Demand Std",
        "Current Inventory",
        "Safety Stock",
        "Reorder Point",
        "Annual Demand",
        "EOQ",
        "Recommended Order Qty"
    ]

    recommendations[numerical_columns] = (
        recommendations[numerical_columns].round(2)
    )

    recommendations.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n==============================")
    print("INVENTORY OPTIMIZATION RESULTS")
    print("==============================")

    print(
        "\nTotal store-product combinations:",
        len(recommendations)
    )

    print("\nAlert Summary:")
    print(
        recommendations["Alert"].value_counts()
    )

    print("\nTop reorder recommendations:")

    reorder = recommendations[
        recommendations["Alert"] == "REORDER"
    ].sort_values(
        "Recommended Order Qty",
        ascending=False
    )

    print(
        reorder.head(15).to_string(index=False)
    )

    print("\nRecommendations saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    generate_inventory_recommendations()