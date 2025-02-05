# Project title
---
## Data-Driven Decision Making in Sales & Marketing
---
### Introduction
In today’s competitive business environment, leveraging data analytics for sales and marketing decisions is crucial for maximizing revenue and optimizing strategies. Companies collect vast amounts of transactional data, but without proper analysis, valuable insights remain untapped. This project applies data-driven decision-making techniques to analyze customer transactions, identify key revenue drivers, and build a predictive model for revenue forecasting.

By examining factors such as customer location, sector performance, pricing impact on sales, and revenue trends, this study provides actionable insights to optimize marketing strategies and sales performance. The predictive model, built using machine learning techniques, forecasts revenue and provides insights for sales growth strategies.

### Project Objectives & Key Research Questions
This project aims to:

- Analyze customer purchase trends across different states and business sectors.

- Identify the relationship between price and sales volume.

- Examine revenue trends over time and determine seasonal effects.

- Build a machine learning model to predict revenue based on key features.

- Evaluate the model’s performance using standard metrics.

- Provide actionable recommendations for marketing and sales optimization.

### Key Research Questions
Based on EDA analysis, the project addresses:

- What are the top 10 states in terms of sales, and how can businesses focus on these regions?

![top_10_states_sales](https://github.com/user-attachments/assets/835953cf-b447-4830-a669-024504d801d0)

This bar chart highlights the top 10 states with the highest sales, helping businesses prioritize marketing and sales strategies in these high-performing regions

- What is the distribution of revenue across all transactions, and are there any patterns or anomalies?

![revenue_distribution](https://github.com/user-attachments/assets/578af4d3-59d8-42d0-b9fb-83980af13e88)

- The histogram illustrates the frequency of revenue values, showing how revenue is distributed across transactions.
- The majority of transactions have lower revenue values, while higher revenue occurrences are less frequent.
- The shape of the distribution helps identify outliers, skewness, or any potential data imbalances in revenue generation.


- What are the seasonal patterns in revenue generation?

![price_vs_sales](https://github.com/user-attachments/assets/5e74a4bc-7f43-4fad-8c58-b0b7f6e21ce2)

This time-series plot illustrates monthly revenue trends over several years, helping businesses identify seasonal peaks and low periods

- Which business sectors contribute most to revenue?

![feature_importance](https://github.com/user-attachments/assets/b19be9e0-5fca-439e-8208-f6b2a7f77f9e)

The feature importance chart shows how various customer state sectors contribute to revenue, guiding businesses in sector-specific marketing efforts.

- How accurately does the model predict revenue, and what are the deviations between actual and predicted values?

![actual_vs_predicted](https://github.com/user-attachments/assets/5b0a4536-6ecb-46d6-92cf-8a11954cd1f5)

- The scatter plot compares actual vs. predicted revenue values to assess the model's accuracy.
- A strong correlation along the diagonal red reference line indicates good predictive performance.
- Deviations from the line highlight cases where the model overestimates or underestimates revenue, guiding potential improvements.

### Description of Dataset, Model, and Algorithms
The dataset, Customer_dataset.csv, consists of transaction records, capturing:

- Year & Month: Time-based sales data.

- Customers_State: The geographical location of customers.

- Sector: The industry sector associated with sales.

- Customers: Number of customers per transaction.

- Price: Product pricing per unit.

- Sales: Sales volume.

- Revenue: Total revenue from sales transactions 

### Data Preprocessing Steps
- Handled missing values, duplicates, and outliers.

- Converted categorical features (state, sector) into numerical form using One-Hot Encoding.

- Standardized numerical features (price, customers, sales) using feature scaling.

- Created new time-based features for trend analysis.

### Model & Algorithms
This project employs Linear Regression, a supervised learning algorithm that models the relationship between independent variables (customer state, sector, price, etc.) and the dependent variable (revenue).

### Define Features and Target

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define target variable
target = "Revenue"  # Change to 'Sales' or another variable if needed

# Define features (excluding the target)
X = df.drop(columns=[target])
y = df[target]

# Identify categorical and numerical columns
categorical_features = ["Customers_State", "Sector", "Year", "Month"]
numerical_features = ["Customers", "Price", "Sales"]

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data Splitting Done: Train Shape:", X_train.shape, "Test Shape:", X_test.shape)
```
| Dataset | Shape        |
|---------|-------------|
| Train   | (34842, 7)  |
| Test    | (8711, 7)   |


### 
```python
# Checking correlation between numerical features
print("\nCorrelation Matrix:\n", df.corr())
```
|            | Customers | Price   | Revenue | Sales   |
|------------|-----------|---------|---------|---------|
| Customers  | 1.000000  | 0.246728 | 0.703906 | 0.601769 |
| Price      | 0.246728  | 1.000000 | 0.120676 | -0.101909 |
| Revenue    | 0.703906  | 0.120676 | 1.000000 | 0.943625 |
| Sales      | 0.601769  | -0.101909 | 0.943625 | 1.000000 |


### Train the Model

```python
# Build a pipeline with preprocessing and regression model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train the model
model.fit(X_train, y_train)

print("Model Training Completed.")
```

### Model Evaluation

```python
# Predict on test data
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation Results:\nMAE: {mae}\nMSE: {mse}\nRMSE: {rmse}\nR² Score: {r2}")
```
| Metric  | Value                |
|---------|----------------------|
| MAE     | 27.9557              |
| MSE     | 1706.0718            |
| RMSE    | 41.3046              |
| R² Score | 0.9659              |


### Findings:

- Sales and Price are the most important features affecting revenue.

- Customer location (state) and sector type significantly impact revenue.

- The model provides accurate revenue predictions with an acceptable R² score.

### Key Graphical Insights

- Revenue Trends Over Time

![revenue_trend](https://github.com/user-attachments/assets/9ce732a8-7936-42bc-96da-ec4e3f0697a7)

📌 Insight: Revenue fluctuates seasonally, indicating peak sales periods.

- Price vs Sales Relationship

![price_vs_sales](https://github.com/user-attachments/assets/9769cd8c-96af-4e75-a46a-817ddf30c138)

📌 Insight: Sales volume decreases as price increases, confirming price sensitivity in customer behavior.

- Top 10 Customer States by Sales
![top_10_states_sales](https://github.com/user-attachments/assets/1703aa7d-2ea1-401c-b7ef-f6dc9f71a394)

📌 Insight: These states drive the majority of total revenue, requiring targeted marketing strategies.



### Interpretation, Discussion, and Conclusion

#### Discussion

- The strong correlation between price and sales suggests that businesses should experiment with dynamic pricing.

- Top states contribute significantly to total revenue, indicating potential regional sales campaigns.

- The Linear Regression model performed well in predicting revenue, but future improvements could involve Random Forest or XGBoost for better accuracy.

#### Conclusion

This study demonstrates how data analytics can enhance business strategies by uncovering sales trends, pricing sensitivity, and regional opportunities. The findings support sales growth through data-driven marketing decisions

### References

This project is based on:


- **Machine Learning Documentation:** [Scikit-learn](https://scikit-learn.org/)
- **Data Visualization:** [Matplotlib](https://matplotlib.org/)
- **Business Analytics Research:** *Harvard Business Review, McKinsey Reports*






