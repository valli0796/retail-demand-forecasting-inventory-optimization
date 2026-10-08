import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

DATA_PATH = "data/processed/sales_data_cleaned.csv"
FORECAST_PATH = "data/processed/forecast_results.csv"
MODEL_PATH = "data/processed/model_comparison.csv"
INVENTORY_PATH = "data/processed/inventory_recommendations.csv"
XGBOOST_PATH = "models/xgboost_demand_model.joblib"
FEATURE_PATH = "models/model_features.joblib"

st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📦",
    layout="wide"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_forecast():
    df = pd.read_csv(FORECAST_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_models():
    return pd.read_csv(MODEL_PATH)


@st.cache_data
def load_inventory():
    return pd.read_csv(INVENTORY_PATH)


@st.cache_resource
def load_xgboost_model():
    model = joblib.load(XGBOOST_PATH)
    features = joblib.load(FEATURE_PATH)
    return model, features


df = load_data()
forecast = load_forecast()
models = load_models()
inventory = load_inventory()

try:
    xgb_model, model_features = load_xgboost_model()
    model_loaded = True
except Exception:
    xgb_model = None
    model_features = []
    model_loaded = False


st.title("📦 Retail Demand Forecasting & Inventory Optimization")

st.caption(
    "AI-powered demand forecasting and inventory decision support system"
)


st.sidebar.header("🔎 Filters")

store_options = ["All"] + sorted(
    df["store_id"].dropna().unique().tolist()
)

selected_store = st.sidebar.selectbox(
    "Store",
    store_options
)

category_options = ["All"] + sorted(
    df["category"].dropna().unique().tolist()
)

selected_category = st.sidebar.selectbox(
    "Category",
    category_options
)

product_options = ["All"] + sorted(
    df["product_id"].dropna().unique().tolist()
)

selected_product = st.sidebar.selectbox(
    "Product",
    product_options
)


filtered_df = df.copy()

if selected_store != "All":
    filtered_df = filtered_df[
        filtered_df["store_id"] == selected_store
    ]

if selected_category != "All":
    filtered_df = filtered_df[
        filtered_df["category"] == selected_category
    ]

if selected_product != "All":
    filtered_df = filtered_df[
        filtered_df["product_id"] == selected_product
    ]


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Overview",
        "📈 Demand Analysis",
        "🔮 Forecasting",
        "📦 Inventory Optimization",
        "🤖 Model Performance"
    ]
)


# =========================================================
# OVERVIEW
# =========================================================

with tab1:

    st.subheader("Business Overview")

    total_demand = filtered_df["demand"].sum()
    total_units_sold = filtered_df["units_sold"].sum()
    average_inventory = filtered_df["inventory_level"].mean()
    total_products = filtered_df["product_id"].nunique()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Demand",
        f"{total_demand:,.0f}"
    )

    col2.metric(
        "Units Sold",
        f"{total_units_sold:,.0f}"
    )

    col3.metric(
        "Avg Inventory",
        f"{average_inventory:,.0f}"
    )

    col4.metric(
        "Products",
        total_products
    )

    st.divider()

    daily_demand = (
        filtered_df
        .groupby("date")["demand"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        daily_demand,
        x="date",
        y="demand",
        title="Daily Demand Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# DEMAND ANALYSIS
# =========================================================

with tab2:

    st.subheader("Demand Analysis")

    col1, col2 = st.columns(2)

    with col1:

        category_demand = (
            filtered_df
            .groupby("category")["demand"]
            .sum()
            .reset_index()
            .sort_values(
                "demand",
                ascending=False
            )
        )

        fig_category = px.bar(
            category_demand,
            x="category",
            y="demand",
            title="Demand by Category"
        )

        st.plotly_chart(
            fig_category,
            use_container_width=True
        )

    with col2:

        region_demand = (
            filtered_df
            .groupby("region")["demand"]
            .sum()
            .reset_index()
            .sort_values(
                "demand",
                ascending=False
            )
        )

        fig_region = px.bar(
            region_demand,
            x="region",
            y="demand",
            title="Demand by Region"
        )

        st.plotly_chart(
            fig_region,
            use_container_width=True
        )

    st.subheader("Promotion Impact")

    promotion_demand = (
        filtered_df
        .groupby("promotion")["demand"]
        .mean()
        .reset_index()
    )

    promotion_demand["promotion"] = (
        promotion_demand["promotion"]
        .map({
            0: "No Promotion",
            1: "Promotion"
        })
    )

    fig_promotion = px.bar(
        promotion_demand,
        x="promotion",
        y="demand",
        title="Average Demand: Promotion vs No Promotion"
    )

    st.plotly_chart(
        fig_promotion,
        use_container_width=True
    )


# =========================================================
# FORECASTING
# =========================================================

with tab3:

    st.subheader("🔮 Demand Forecasting")

    if model_loaded:
        st.success(
            "XGBoost model loaded successfully from the saved model file."
        )
    else:
        st.error(
            "XGBoost model could not be loaded. "
            "Run src/train_model.py first."
        )

    best_model = models.iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Selected Model",
        best_model["Model"]
    )

    col2.metric(
        "MAE",
        f"{best_model['MAE']:.2f}"
    )

    col3.metric(
        "MAPE",
        f"{best_model['MAPE']:.2f}%"
    )

    st.divider()

    forecast_chart = px.line(
        forecast,
        x="date",
        y=["demand", "Forecast"],
        title="Actual vs XGBoost Forecast"
    )

    st.plotly_chart(
        forecast_chart,
        use_container_width=True
    )

    st.subheader("Forecast Results")

    forecast_display = forecast.copy()

    forecast_display["Error"] = (
        forecast_display["demand"]
        - forecast_display["Forecast"]
    )

    st.dataframe(
        forecast_display,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# INVENTORY OPTIMIZATION
# =========================================================

with tab4:

    st.subheader("📦 Inventory Optimization")

    reorder = inventory[
        inventory["Alert"] == "REORDER"
    ]

    low_stock = inventory[
        inventory["Alert"] == "LOW STOCK"
    ]

    normal = inventory[
        inventory["Alert"] == "NORMAL"
    ]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "🔴 Reorder",
        len(reorder)
    )

    col2.metric(
        "🟡 Low Stock",
        len(low_stock)
    )

    col3.metric(
        "🟢 Normal",
        len(normal)
    )

    st.divider()

    alert_counts = (
        inventory["Alert"]
        .value_counts()
        .reset_index()
    )

    alert_counts.columns = [
        "Alert",
        "Count"
    ]

    fig_alert = px.bar(
        alert_counts,
        x="Alert",
        y="Count",
        title="Inventory Risk Distribution"
    )

    st.plotly_chart(
        fig_alert,
        use_container_width=True
    )

    st.subheader("🚨 Reorder Recommendations")

    reorder_display = reorder.sort_values(
        "Current Inventory"
    )

    st.dataframe(
        reorder_display,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Complete Inventory Recommendations")

    st.dataframe(
        inventory.sort_values(
            "Current Inventory"
        ),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

with tab5:

    st.subheader("🤖 Forecasting Model Comparison")

    st.dataframe(
        models,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    fig_mae = px.bar(
        models,
        x="Model",
        y="MAE",
        title="MAE Comparison"
    )

    st.plotly_chart(
        fig_mae,
        use_container_width=True
    )

    fig_rmse = px.bar(
        models,
        x="Model",
        y="RMSE",
        title="RMSE Comparison"
    )

    st.plotly_chart(
        fig_rmse,
        use_container_width=True
    )

    fig_mape = px.bar(
        models,
        x="Model",
        y="MAPE",
        title="MAPE Comparison"
    )

    st.plotly_chart(
        fig_mape,
        use_container_width=True
    )

    st.success(
        f"Selected Model: {best_model['Model']} | "
        f"MAE: {best_model['MAE']:.2f} | "
        f"RMSE: {best_model['RMSE']:.2f} | "
        f"MAPE: {best_model['MAPE']:.2f}%"
    )


st.divider()

st.caption(
    "Retail Demand Forecasting & Inventory Optimization | "
    "Python • SQL • Statistics • XGBoost • Streamlit"
)