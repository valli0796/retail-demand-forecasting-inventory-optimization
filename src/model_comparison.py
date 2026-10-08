import pandas as pd

results = [
    {
        "Model": "Naive Forecast",
        "MAE": 1266.953947,
        "RMSE": 1862.373438,
        "MAPE": 15.887984
    },
    {
        "Model": "Exponential Smoothing",
        "MAE": 1310.047898,
        "RMSE": 1862.004792,
        "MAPE": 16.094003
    },
    {
        "Model": "ARIMA",
        "MAE": 1426.411987,
        "RMSE": 2095.352855,
        "MAPE": 18.105278
    },
    {
        "Model": "XGBoost",
        "MAE": 234.77,
        "RMSE": 293.45,
        "MAPE": 2.49
    }
]

comparison = pd.DataFrame(results)

comparison = comparison.sort_values("MAE").reset_index(drop=True)

print("\n==============================")
print("FINAL MODEL COMPARISON")
print("==============================")

print(comparison.to_string(index=False))

best_model = comparison.iloc[0]

print("\n==============================")
print("BEST MODEL")
print("==============================")
print("Model:", best_model["Model"])
print("MAE:", best_model["MAE"])
print("RMSE:", best_model["RMSE"])
print("MAPE:", best_model["MAPE"])

comparison.to_csv(
    "data/processed/model_comparison.csv",
    index=False
)

print("\nComparison saved to:")
print("data/processed/model_comparison.csv")