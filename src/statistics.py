import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "retail_sales_cleaned.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def promotion_analysis(df):
    print("\n--- Promotion Analysis ---")

    promotion_demand = df.groupby("promotion")["demand"].agg(
        ["count", "mean", "sum"]
    )

    print(promotion_demand)

    groups = [
        group["demand"].values
        for _, group in df.groupby("promotion")
    ]

    if len(groups) == 2:
        statistic, p_value = stats.ttest_ind(
            groups[0],
            groups[1],
            equal_var=False
        )

        print("\nT-test statistic:", statistic)
        print("P-value:", p_value)

        if p_value < 0.05:
            print("Result: Promotion has a statistically significant effect on demand.")
        else:
            print("Result: No statistically significant promotion effect detected.")


def price_demand_correlation(df):
    print("\n--- Price-Demand Correlation ---")

    correlation, p_value = stats.pearsonr(
        df["price"],
        df["demand"]
    )

    print("Correlation:", correlation)
    print("P-value:", p_value)

    if p_value < 0.05:
        print("Result: Price and demand have a statistically significant relationship.")
    else:
        print("Result: No statistically significant relationship detected.")


def discount_demand_correlation(df):
    print("\n--- Discount-Demand Correlation ---")

    correlation, p_value = stats.pearsonr(
        df["discount"],
        df["demand"]
    )

    print("Correlation:", correlation)
    print("P-value:", p_value)

    if p_value < 0.05:
        print("Result: Discount and demand have a statistically significant relationship.")
    else:
        print("Result: No statistically significant relationship detected.")


def seasonality_analysis(df):
    print("\n--- Seasonality Analysis ---")

    daily_demand = (
        df.groupby("date")["demand"]
        .sum()
        .sort_index()
    )

    print("\nDaily demand:")
    print(daily_demand.head())

    if len(daily_demand) >= 14:
        decomposition = seasonal_decompose(
            daily_demand,
            model="additive",
            period=7
        )

        print("\nSeasonality analysis completed.")
        print("Trend sample:")
        print(decomposition.trend.dropna().head())

        print("\nSeasonal component sample:")
        print(decomposition.seasonal.head())


def inventory_demand_correlation(df):
    print("\n--- Inventory-Demand Correlation ---")

    correlation, p_value = stats.pearsonr(
        df["inventory_level"],
        df["demand"]
    )

    print("Correlation:", correlation)
    print("P-value:", p_value)

    if p_value < 0.05:
        print("Result: Inventory level and demand have a statistically significant relationship.")
    else:
        print("Result: No statistically significant relationship detected.")


def main():
    df = load_data()

    print("Dataset loaded successfully")
    print("Rows:", len(df))

    promotion_analysis(df)
    price_demand_correlation(df)
    discount_demand_correlation(df)
    inventory_demand_correlation(df)
    seasonality_analysis(df)


if __name__ == "__main__":
    main()