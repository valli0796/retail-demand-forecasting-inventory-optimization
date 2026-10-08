import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📦",
    layout="wide"
)

DATA_PATH = "data/processed/retail_sales_cleaned.csv"
FORECAST_PATH = "data/processed/product_forecast_results.csv"
INVENTORY_PATH = "data/processed/forecast_inventory_recommendations.csv"
MODEL_PATH = "models/xgboost_demand_model.joblib"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_forecasts():
    if os.path.exists(FORECAST_PATH):
        df = pd.read_csv(FORECAST_PATH)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


@st.cache_data
def load_inventory():
    if os.path.exists(INVENTORY_PATH):
        return pd.read_csv(INVENTORY_PATH)
    return pd.DataFrame()


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


df = load_data()
forecast_df = load_forecasts()
inventory_df = load_inventory()
model = load_model()


st.title(
    "📦 Retail Demand Forecasting & Inventory Optimization"
)

st.markdown(
    "### Forecast demand. Optimize inventory. Make better replenishment decisions."
)


if model is not None:
    st.success(
        "XGBoost demand forecasting model loaded successfully."
    )


# ==========================================================
# SIDEBAR FILTERS
# ==========================================================

st.sidebar.header("🔎 Filters")

stores = sorted(
    df["store_id"].unique()
)

products = sorted(
    df["product_id"].unique()
)

categories = sorted(
    df["category"].unique()
)

selected_store = st.sidebar.selectbox(
    "Store",
    ["All"] + stores
)

selected_product = st.sidebar.selectbox(
    "Product",
    ["All"] + products
)

selected_category = st.sidebar.selectbox(
    "Category",
    ["All"] + categories
)


filtered_df = df.copy()


if selected_store != "All":
    filtered_df = filtered_df[
        filtered_df["store_id"] == selected_store
    ]


if selected_product != "All":
    filtered_df = filtered_df[
        filtered_df["product_id"] == selected_product
    ]


if selected_category != "All":
    filtered_df = filtered_df[
        filtered_df["category"] == selected_category
    ]


# ==========================================================
# TABS
# ==========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Overview",
        "📈 Demand Analysis",
        "🔮 Forecasting",
        "📦 Inventory Optimization",
        "🤖 Model Performance"
    ]
)


# ==========================================================
# TAB 1 - OVERVIEW
# ==========================================================

with tab1:

    st.header("Business Overview")

    total_demand = filtered_df["demand"].sum()

    total_units = filtered_df["units_sold"].sum()

    average_demand = filtered_df["demand"].mean()

    average_inventory = filtered_df[
        "inventory_level"
    ].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Demand",
        f"{total_demand:,.0f}"
    )

    col2.metric(
        "Units Sold",
        f"{total_units:,.0f}"
    )

    col3.metric(
        "Average Demand",
        f"{average_demand:,.2f}"
    )

    col4.metric(
        "Average Inventory",
        f"{average_inventory:,.2f}"
    )

    st.subheader("Demand Trend")

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
        title="Daily Demand"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================================
# TAB 2 - DEMAND ANALYSIS
# ==========================================================

with tab2:

    st.header("Demand Analysis")

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

        fig = px.bar(
            category_demand,
            x="category",
            y="demand",
            title="Demand by Category"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

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

        fig = px.bar(
            promotion_demand,
            x="promotion",
            y="demand",
            title="Average Demand: Promotion vs No Promotion"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Top Products by Demand")

    top_products = (
        filtered_df
        .groupby("product_id")["demand"]
        .sum()
        .reset_index()
        .sort_values(
            "demand",
            ascending=False
        )
        .head(10)
    )

    fig = px.bar(
        top_products,
        x="product_id",
        y="demand",
        title="Top 10 Products"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================================
# TAB 3 - FORECASTING
# ==========================================================

with tab3:

    st.header(
        "🔮 Product-Level Demand Forecasting"
    )

    if forecast_df.empty:

        st.warning(
            "Product forecast file not found. "
            "Run product_forecasting.py first."
        )

    else:

        forecast_filtered = forecast_df.copy()

        if selected_store != "All":
            forecast_filtered = forecast_filtered[
                forecast_filtered["store_id"]
                == selected_store
            ]

        if selected_product != "All":
            forecast_filtered = forecast_filtered[
                forecast_filtered["product_id"]
                == selected_product
            ]

        st.metric(
            "Forecast Records",
            f"{len(forecast_filtered):,}"
        )

        if not forecast_filtered.empty:

            st.subheader(
                "Actual vs Forecast Demand"
            )

            forecast_plot = forecast_filtered.copy()

            if (
                selected_store == "All"
                and selected_product == "All"
            ):

                daily_forecast = (
                    forecast_plot
                    .groupby("date")
                    .agg({
                        "demand": "sum",
                        "Forecast": "sum"
                    })
                    .reset_index()
                )

                fig = px.line(
                    daily_forecast,
                    x="date",
                    y=[
                        "demand",
                        "Forecast"
                    ],
                    title="Actual vs Forecast Demand"
                )

            else:

                fig = px.line(
                    forecast_plot,
                    x="date",
                    y=[
                        "demand",
                        "Forecast"
                    ],
                    title="Actual vs Forecast Demand"
                )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.subheader("Forecast Sample")

            st.dataframe(
                forecast_filtered.head(50),
                use_container_width=True,
                hide_index=True
            )


# ==========================================================
# TAB 4 - INVENTORY OPTIMIZATION
# ==========================================================

with tab4:

    st.header(
        "📦 Inventory Optimization"
    )

    if inventory_df.empty:

        st.warning(
            "Inventory recommendation file not found. "
            "Run forecast_inventory.py first."
        )

    else:

        inventory_filtered = inventory_df.copy()

        if selected_store != "All":

            inventory_filtered = inventory_filtered[
                inventory_filtered["store_id"]
                == selected_store
            ]

        if selected_product != "All":

            inventory_filtered = inventory_filtered[
                inventory_filtered["product_id"]
                == selected_product
            ]

        # --------------------------------------------------
        # INVENTORY SUMMARY
        # --------------------------------------------------

        total_products = len(
            inventory_filtered
        )

        reorder_count = len(
            inventory_filtered[
                inventory_filtered["Alert"]
                == "REORDER"
            ]
        )

        low_stock_count = len(
            inventory_filtered[
                inventory_filtered["Alert"]
                == "LOW STOCK"
            ]
        )

        normal_count = len(
            inventory_filtered[
                inventory_filtered["Alert"]
                == "NORMAL"
            ]
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Products",
            total_products
        )

        col2.metric(
            "🔴 Reorder",
            reorder_count
        )

        col3.metric(
            "🟡 Low Stock",
            low_stock_count
        )

        col4.metric(
            "🟢 Normal",
            normal_count
        )

        # --------------------------------------------------
        # ALERT DISTRIBUTION
        # --------------------------------------------------

        st.subheader(
            "Inventory Status Distribution"
        )

        alert_summary = (
            inventory_filtered[
                "Alert"
            ]
            .value_counts()
            .reset_index()
        )

        alert_summary.columns = [
            "Alert",
            "Count"
        ]

        fig = px.pie(
            alert_summary,
            names="Alert",
            values="Count",
            title="Inventory Alert Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # --------------------------------------------------
        # PRODUCT DRILL-DOWN
        # --------------------------------------------------

        st.subheader(
            "🔎 Product Inventory Drill-Down"
        )

        available_stores = sorted(
            inventory_df["store_id"]
            .unique()
        )

        drill_store = st.selectbox(
            "Select Store",
            available_stores,
            key="inventory_store"
        )

        store_products = sorted(
            inventory_df[
                inventory_df["store_id"]
                == drill_store
            ]["product_id"]
            .unique()
        )

        drill_product = st.selectbox(
            "Select Product",
            store_products,
            key="inventory_product"
        )

        selected_inventory = inventory_df[
            (
                inventory_df["store_id"]
                == drill_store
            )
            &
            (
                inventory_df["product_id"]
                == drill_product
            )
        ]

        if not selected_inventory.empty:

            row = selected_inventory.iloc[0]

            st.markdown(
                f"### {drill_store} — {drill_product}"
            )

            alert = row["Alert"]

            if alert == "REORDER":

                st.error(
                    "🔴 REORDER REQUIRED"
                )

            elif alert == "LOW STOCK":

                st.warning(
                    "🟡 LOW STOCK"
                )

            else:

                st.success(
                    "🟢 INVENTORY LEVEL NORMAL"
                )

            # --------------------------------------------------
            # KEY INVENTORY METRICS
            # --------------------------------------------------

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Current Inventory",
                f"{row['Current Inventory']:,.0f}"
            )

            col2.metric(
                "Forecast Demand",
                f"{row['Average Forecast Demand']:,.2f}"
            )

            col3.metric(
                "Safety Stock",
                f"{row['Safety Stock']:,.2f}"
            )

            col4.metric(
                "Reorder Point",
                f"{row['Reorder Point']:,.2f}"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "EOQ",
                f"{row['EOQ']:,.2f}"
            )

            col2.metric(
                "Recommended Order",
                f"{row['Recommended Order Qty']:,.2f}"
            )

            col3.metric(
                "Annual Demand",
                f"{row['Annual Demand']:,.0f}"
            )

            # --------------------------------------------------
            # INVENTORY POSITION
            # --------------------------------------------------

            st.subheader(
                "Inventory Position"
            )

            inventory_chart = pd.DataFrame({
                "Metric": [
                    "Current Inventory",
                    "Safety Stock",
                    "Reorder Point"
                ],
                "Units": [
                    row["Current Inventory"],
                    row["Safety Stock"],
                    row["Reorder Point"]
                ]
            })

            fig = px.bar(
                inventory_chart,
                x="Metric",
                y="Units",
                title=(
                    f"Inventory Position — "
                    f"{drill_store} / {drill_product}"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # --------------------------------------------------
            # FORECAST TREND
            # --------------------------------------------------

            st.subheader(
                "Demand Forecast Trend"
            )

            product_forecast = forecast_df[
                (
                    forecast_df["store_id"]
                    == drill_store
                )
                &
                (
                    forecast_df["product_id"]
                    == drill_product
                )
            ].copy()

            if not product_forecast.empty:

                fig = px.line(
                    product_forecast,
                    x="date",
                    y=[
                        "demand",
                        "Forecast"
                    ],
                    title=(
                        f"Actual vs Forecast Demand — "
                        f"{drill_store} / {drill_product}"
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            # --------------------------------------------------
            # INVENTORY DECISION DETAILS
            # --------------------------------------------------

            st.subheader(
                "Inventory Decision Details"
            )

            details = pd.DataFrame({
                "Metric": [
                    "Store",
                    "Product",
                    "Average Forecast Demand",
                    "Demand Standard Deviation",
                    "Current Inventory",
                    "Safety Stock",
                    "Reorder Point",
                    "Annual Demand",
                    "EOQ",
                    "Recommended Order Quantity",
                    "Inventory Alert"
                ],
                "Value": [
                    row["store_id"],
                    row["product_id"],
                    row["Average Forecast Demand"],
                    row["Demand Std"],
                    row["Current Inventory"],
                    row["Safety Stock"],
                    row["Reorder Point"],
                    row["Annual Demand"],
                    row["EOQ"],
                    row["Recommended Order Qty"],
                    row["Alert"]
                ]
            })

            st.dataframe(
                details,
                use_container_width=True,
                hide_index=True
            )

        # --------------------------------------------------
        # REORDER RECOMMENDATIONS
        # --------------------------------------------------

        st.subheader(
            "🚨 Reorder Recommendations"
        )

        reorder_table = inventory_filtered[
            inventory_filtered["Alert"]
            == "REORDER"
        ].sort_values(
            "Recommended Order Qty",
            ascending=False
        )

        if reorder_table.empty:

            st.success(
                "No products currently require reordering."
            )

        else:

            st.dataframe(
                reorder_table,
                use_container_width=True,
                hide_index=True
            )

        # --------------------------------------------------
        # ALL INVENTORY RECOMMENDATIONS
        # --------------------------------------------------

        st.subheader(
            "All Inventory Recommendations"
        )

        st.dataframe(
            inventory_filtered,
            use_container_width=True,
            hide_index=True
        )


# ==========================================================
# TAB 5 - MODEL PERFORMANCE
# ==========================================================

with tab5:

    st.header(
        "🤖 Model Performance"
    )

    comparison_path = (
        "data/processed/model_comparison.csv"
    )

    if os.path.exists(
        comparison_path
    ):

        comparison = pd.read_csv(
            comparison_path
        )

        st.subheader(
            "Forecasting Model Comparison"
        )

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(
            comparison,
            x="Model",
            y="MAE",
            title="Model Comparison - MAE"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        fig = px.bar(
            comparison,
            x="Model",
            y="RMSE",
            title="Model Comparison - RMSE"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        fig = px.bar(
            comparison,
            x="Model",
            y="MAPE",
            title="Model Comparison - MAPE"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Model comparison file not found."
        )

