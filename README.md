Title:Retail Demand Forecasting & Inventory Optimization

A data science and machine learning project that analyzes historical retail sales data, forecasts future demand, and generates inventory optimization recommendations such as safety stock, reorder points, and Economic Order Quantity (EOQ).

2.Project Overview

Retail businesses need accurate demand forecasts to maintain sufficient inventory while avoiding overstocking.

This project combines:

Data Cleaning → EDA → SQL → Statistical Analysis → Time-Series Forecasting → XGBoost → Inventory Optimization → Dashboard → Docker

The system analyzes historical retail demand and uses machine learning to predict future demand at both aggregate and product levels. The forecasts are then used to support inventory decisions.

3.Objectives

* Clean and prepare retail sales data
* Perform exploratory data analysis
* Analyze demand using SQL
* Identify trends, seasonality, and relationships between variables
* Compare traditional forecasting models with machine learning
* Forecast future retail demand
* Calculate safety stock and reorder points
* Calculate Economic Order Quantity (EOQ)
* Identify products requiring inventory replenishment
* Provide an interactive Streamlit dashboard
* Containerize the application using Docker
* Add automated tests using Pytest

4.Dataset

The project uses historical retail sales data containing information about:

* Date
* Store
* Product
* Category
* Region
* Inventory level
* Units sold
* Units ordered
* Price
* Discount
* Weather condition
* Promotion
* Competitor pricing
* Seasonality
* Epidemic indicator
* Demand

The cleaned dataset contains approximately 76,000 records.

5.Project Architecture

Retail Sales Data
       |
       v
Data Cleaning & Preprocessing
       |
       v
Exploratory Data Analysis
       |
       +---------> PostgreSQL / SQL Analysis
       |
       +---------> Statistical Analysis
       |
       v
Feature Engineering
       |
       v
Demand Forecasting
       |
       +---- Naive Forecast
       +---- Exponential Smoothing
       +---- ARIMA
       +---- XGBoost
       |
       v
Forecast Evaluation
       |
       v
Inventory Optimization
       |
       +---- Safety Stock
       +---- Reorder Point
       +---- EOQ
       +---- Order Recommendation
       |
       v
Streamlit Dashboard
       |
       v
Docker Deployment

6.Machine Learning

XGBoost Demand Forecasting
The final machine learning model uses historical demand patterns and external/business features.

Features include:

* Lag demand: 1, 7, 14, 28 days
* Rolling mean: 7, 14, 28 days
* Rolling standard deviation
* Inventory level
* Price
* Discount
* Competitor pricing
* Promotion
* Epidemic indicator
* Month
* Week of year
* Day
* Day of week
* Weekend indicator

Chronological train-test splitting was used to avoid future-data leakage.

7.Aggregate Forecasting Performance

| Model                 |        MAE |       RMSE |      MAPE |
| --------------------- | ---------: | ---------: | --------: |
| Naive Forecast        |    1266.95 |    1862.37 |    15.89% |
| Exponential Smoothing |    1310.05 |    1862.00 |    16.09% |
| ARIMA                 |    1426.41 |    2095.35 |    18.11% |
| XGBoost           | 234.77 | 293.45 | 2.49% |

On this dataset, XGBoost achieved the best forecasting performance among the evaluated models.

8.Product-Level Forecasting

The model was also evaluated at the store-product level.

* Store-product combinations: 100
* Training records: 58,600
* Testing records: 14,600
* MAE: 25.96
* RMSE: 33.84
* MAPE: 37.2%

The higher product-level MAPE reflects the greater variability of individual product demand and the sensitivity of MAPE to smaller demand values.

9.Statistical Analysis

Statistical analysis was performed to understand relationships in the retail data.

10.Promotion and Demand

Average demand:

* Without promotion: 95.03
* With promotion: 123.27

Demand was approximately 29.7% higher during promotions.

A Welch's t-test showed a statistically significant difference.

11.Price and Demand

Pearson correlation:
Correlation = -0.0235

The relationship was statistically significant but practically very weak.

12.Discount and Demand

Correlation = 0.2247
This indicates a positive relationship between discount levels and demand in the dataset.

13.Inventory and Demand

Correlation = 0.1266
This represents a weak positive relationship.

Weekly seasonality was also identified through seasonal decomposition.

14.Inventory Optimization

The forecasting results are converted into inventory decisions.

15.Safety Stock

Safety stock is calculated using:
Safety Stock = Z × Demand Standard Deviation × √Lead Time

The project uses:

Service Level = 95%
Z Score = 1.645
Lead Time = 7 days

16.Reorder Point

Reorder Point =
Average Daily Demand × Lead Time + Safety Stock


17.Economic Order Quantity

EOQ is calculated using:
EOQ = √((2 × Annual Demand × Ordering Cost) / Holding Cost)

Assumptions:

Ordering Cost = 100
Holding Cost = 5

The system generates:

* Current inventory
* Average demand
* Demand variability
* Safety stock
* Reorder point
* Annual demand
* EOQ
* Recommended order quantity
* Inventory alert

Possible alerts:

REORDER
LOW STOCK
NORMAL

18.SQL Analysis

PostgreSQL is used for business-oriented analysis.

Examples include:

* Total sales and demand
* Top-performing products
* Category performance
* Regional performance
* Store performance
* Promotion vs non-promotion demand
* Inventory risk
* Monthly demand
* Weekly demand
* Weekend vs weekday demand
* Price vs demand
* Discount vs demand
* Weather vs demand
* High-demand products

19.Dashboard

The project includes an interactive Streamlit dashboard.

20.Dashboard Sections

1. Overview
2. Demand Analysis
3. Forecasting
4. Inventory Optimization
5. Model Performance

The dashboard supports filtering by:

* Store
* Product
* Category

It provides visualizations for:

* Demand trends
* Actual vs forecast demand
* Inventory position
* Forecast performance
* Inventory alerts
* Reorder recommendations
* Model comparison

21.Testing

Pytest is used for testing the forecasting and inventory logic.

Current test result:
10 passed

Tests cover:

* MAPE calculation
* Perfect forecast
* Non-negative forecasts
* Safety stock
* Demand variability
* Reorder point
* EOQ
* Zero-demand EOQ
* EOQ behavior with changing demand

22.Docker

The Streamlit dashboard is containerized using Docker.

Build the image:

docker build -t retail-demand-forecasting .

Run the application:

docker run -p 8501:8501 retail-demand-forecasting

Then open:

http://localhost:8501


23.Project Structure

Retail_Demand_Forecasting/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_data_cleaning.ipynb
│
├── src/
│   ├── database.py
│   ├── forecasting.py
│   ├── forecast_inventory.py
│   ├── inventory.py
│   ├── model_comparison.py
│   ├── product_forecasting.py
│   ├── statistics.py
│   └── train_model.py
│
├── sql/
│   ├── schema.sql
│   └── analysis_queries.sql
│
├── models/
│   ├── model_features.joblib
│   └── xgboost_demand_model.joblib
│
├── dashboard/
│   └── app.py
│
├── tests/
│   ├── test_forecasting.py
│   └── test_inventory.py
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── main.py
├── requirements.txt
└── README.md


24.Technology Stack

Programming

* Python

Data Science

* Pandas
* NumPy
* SciPy
* Scikit-learn

Machine Learning

* XGBoost
* Time-Series Forecasting
* Feature Engineering
* Model Evaluation

Statistics

* Welch's t-test
* Pearson Correlation
* Seasonal Decomposition

Database

* PostgreSQL
* SQL
* SQLAlchemy
* psycopg2

Visualization

* Matplotlib
* Seaborn
* Plotly

Dashboard

* Streamlit

Testing

* Pytest

Deployment & Development

* Docker
* Git
* GitHub
* Joblib

How to Run Locally

Clone the repository:

git clone https://github.com/valli0796/retail-demand-forecasting-inventory-optimization.git
cd retail-demand-forecasting-inventory-optimization

Create a virtual environment:
python -m venv venv

Activate it on Windows:
.\venv\Scripts\Activate.ps1

Install dependencies:
pip install -r requirements.txt

Run inventory optimization:
python src/forecast_inventory.py

Run tests:
pytest

Run the dashboard:
streamlit run dashboard/app.py

25.Key Results

* 76,000+ retail records analyzed
* XGBoost aggregate forecast MAPE: 2.49%
* XGBoost aggregate MAE: 234.77
* XGBoost aggregate RMSE: 293.45
* 100 store-product combinations analyzed
* 10/10 automated tests passed
* Interactive Streamlit dashboard
* PostgreSQL-based SQL analysis
* Dockerized application

26.Business Value

This project demonstrates how historical retail data can be transformed into actionable business decisions.

Instead of only predicting demand, the system connects forecasting with inventory management by answering:

> How much demand should we expect, when should we reorder, and how much should we order?

This helps businesses reduce stockout risk, maintain appropriate safety stock, and make data-driven inventory decisions.

27.Future Improvements

* Real-time sales data integration
* Automated model retraining
* Hyperparameter optimization
* Probabilistic forecasting
* Multi-step future forecasting
* Cost-aware inventory optimization
* Supplier-specific lead times
* Real-time inventory alerts
* Cloud deployment
* Automated CI/CD pipeline
* Advanced forecasting models such as LightGBM and Temporal Fusion Transformers


