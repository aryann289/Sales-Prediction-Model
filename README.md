# Large-Scale Sales Prediction Using Machine Learning

This project implements a **large-scale sales prediction system** using multiple machine learning algorithms to forecast sales based on historical retail data.  
The goal is to compare different regression models and identify the most accurate approach for real-world sales forecasting.

---

# Project Overview

Sales forecasting is a critical business task for inventory planning, demand forecasting, and revenue optimization.  
In this project, we build and evaluate multiple ML models to predict **Sales** using structured retail transaction data.

# Key Highlights
- Handles **large datasets**
- Uses **multiple machine learning algorithms**
- Performs **feature engineering on date & categorical data**
- Compares models using standard regression metrics
- Follows **industry-standard ML workflow**

---

# Dataset

**Dataset Name:** `stores_sales.csv`

# Key Columns:
- `Order Date` – Date of purchase
- `Category`, `Sub-Category` – Product information
- `Region`, `City`, `State` – Geographical data
- `Discount`, `Quantity` – Pricing & demand indicators
- `Sales` – **Target variable**

> Dataset contains real-world retail transactions with time-based patterns.

---

# Machine Learning Models Used

The following regression models are trained and evaluated:

1. **Linear Regression**
2. **Ridge Regression**
3. **Lasso Regression**
4. **Random Forest Regressor**
5. **Gradient Boosting Regressor**
6. **XGBoost Regressor**

Each model is trained on the same feature set and evaluated using identical metrics.

---

# Feature Engineering

- Extracted date features:
  - Year
  - Month
  - Day
  - Weekday
- Encoded categorical variables using `LabelEncoder`
- Removed high-cardinality and non-informative columns
- Applied time-aware train-test split to prevent data leakage

---

# Model Evaluation Metrics

Each model is evaluated using:

- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- **R² Score**

The best-performing model is selected based on **lowest RMSE** and **highest R²**.

---

# Tech Stack

- **Language:** Python
- **Libraries:**
  - pandas
  - numpy
  - scikit-learn
  - xgboost
- **IDE:** VS Code
