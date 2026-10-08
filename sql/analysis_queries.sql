-- ============================================================
-- RETAIL DEMAND FORECASTING & INVENTORY OPTIMIZATION
-- SQL BUSINESS ANALYSIS
-- ============================================================


-- 1. Total number of records
SELECT COUNT(*) AS total_records
FROM retail_sales;


-- 2. Total units sold
SELECT
    SUM(units_sold) AS total_units_sold
FROM retail_sales;


-- 3. Total demand
SELECT
    SUM(demand) AS total_demand
FROM retail_sales;


-- 4. Average daily demand
SELECT
    AVG(demand) AS average_demand
FROM retail_sales;


-- 5. Top 10 products by total demand
SELECT
    product_id,
    SUM(demand) AS total_demand
FROM retail_sales
GROUP BY product_id
ORDER BY total_demand DESC
LIMIT 10;


-- 6. Demand by category
SELECT
    category,
    SUM(demand) AS total_demand,
    AVG(demand) AS average_demand
FROM retail_sales
GROUP BY category
ORDER BY total_demand DESC;


-- 7. Demand by region
SELECT
    region,
    SUM(demand) AS total_demand,
    AVG(demand) AS average_demand
FROM retail_sales
GROUP BY region
ORDER BY total_demand DESC;


-- 8. Store performance
SELECT
    store_id,
    SUM(units_sold) AS total_units_sold,
    SUM(demand) AS total_demand,
    AVG(demand) AS average_demand
FROM retail_sales
GROUP BY store_id
ORDER BY total_demand DESC;


-- 9. Promotion vs non-promotion demand
SELECT
    promotion,
    COUNT(*) AS records,
    AVG(demand) AS average_demand,
    SUM(demand) AS total_demand
FROM retail_sales
GROUP BY promotion
ORDER BY average_demand DESC;


-- 10. Units ordered vs units sold
SELECT
    SUM(units_ordered) AS total_units_ordered,
    SUM(units_sold) AS total_units_sold,
    SUM(units_ordered) - SUM(units_sold) AS difference
FROM retail_sales;


-- 11. Products with low inventory
SELECT
    product_id,
    AVG(inventory_level) AS average_inventory,
    SUM(demand) AS total_demand
FROM retail_sales
GROUP BY product_id
ORDER BY average_inventory ASC
LIMIT 10;


-- 12. High-demand products
SELECT
    product_id,
    SUM(demand) AS total_demand,
    AVG(inventory_level) AS average_inventory
FROM retail_sales
GROUP BY product_id
ORDER BY total_demand DESC
LIMIT 10;


-- 13. Monthly demand
SELECT
    year,
    month,
    SUM(demand) AS total_demand
FROM retail_sales
GROUP BY year, month
ORDER BY year, month;


-- 14. Demand by day of week
SELECT
    day_of_week,
    AVG(demand) AS average_demand,
    SUM(demand) AS total_demand
FROM retail_sales
GROUP BY day_of_week
ORDER BY day_of_week;


-- 15. Weekend vs weekday demand
SELECT
    is_weekend,
    AVG(demand) AS average_demand,
    SUM(demand) AS total_demand
FROM retail_sales
GROUP BY is_weekend
ORDER BY is_weekend;


-- 16. Average demand by seasonality
SELECT
    seasonality,
    AVG(demand) AS average_demand,
    SUM(demand) AS total_demand
FROM retail_sales
GROUP BY seasonality
ORDER BY total_demand DESC;


-- 17. Price vs demand
SELECT
    AVG(price) AS average_price,
    AVG(demand) AS average_demand
FROM retail_sales;


-- 18. Discount vs demand
SELECT
    AVG(discount) AS average_discount,
    AVG(demand) AS average_demand
FROM retail_sales;


-- 19. Weather condition and demand
SELECT
    weather_condition,
    AVG(demand) AS average_demand,
    SUM(demand) AS total_demand
FROM retail_sales
GROUP BY weather_condition
ORDER BY average_demand DESC;


-- 20. Inventory risk
SELECT
    product_id,
    AVG(inventory_level) AS average_inventory,
    AVG(demand) AS average_demand
FROM retail_sales
GROUP BY product_id
HAVING AVG(inventory_level) < AVG(demand)
ORDER BY average_inventory ASC;