import os
import pandas as pd
from sqlalchemy import create_engine


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:Valli1602!@localhost:5432/retail_demand_db"
)


def get_engine():
    return create_engine(DATABASE_URL)


def load_data_to_postgresql():
    csv_path = "data/processed/sales_data_cleaned.csv"

    df = pd.read_csv(csv_path)

    print("CSV loaded successfully")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    engine = get_engine()

    df.to_sql(
        "retail_sales",
        engine,
        if_exists="append",
        index=False
    )

    print("Data loaded successfully into PostgreSQL")


if __name__ == "__main__":
    load_data_to_postgresql()