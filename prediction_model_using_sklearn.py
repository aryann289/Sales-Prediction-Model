# 1. Import Required Libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor


# 1. Load Dataset

data = pd.read_csv(
    "stores_sales_forecasting.csv",
    encoding="latin1"
)


# 2. Date Parsing

data["Order Date"] = pd.to_datetime(data["Order Date"], errors="coerce")

data["Year"] = data["Order Date"].dt.year
data["Month"] = data["Order Date"].dt.month
data["Day"] = data["Order Date"].dt.day
data["Weekday"] = data["Order Date"].dt.weekday


# 3. Drop Unnecessary Columns

drop_cols = [
    "Row ID",
    "Order ID",
    "Customer Name",
    "Product Name",
    "Ship Date",
    "Order Date"
]

data.drop(columns=drop_cols, inplace=True, errors="ignore")


# 4. Encode Categorical Columns (SAFE)

for col in data.select_dtypes(include="object").columns:
    data[col] = LabelEncoder().fit_transform(data[col].astype(str))


# 5. Define X and y

X = data.drop("Sales", axis=1)
y = data["Sales"]


# 6. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)


# 7. Models

models = {
    "Linear Regression": LinearRegression(),
    
    "Ridge Regression": Ridge(alpha=1.0),
    
    "Lasso Regression": Lasso(alpha=0.01),
    
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        n_jobs=-1,
        random_state=42
    ),
    
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    ),
    
    "XGBoost": XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42
    )
}

# 8. Train & Evaluate

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    results.append([name, mae, rmse, r2])
    
    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2  : {r2:.4f}")

# 10. Model Comparison Table

results_df = pd.DataFrame(
    results,
    columns=["Model", "MAE", "RMSE", "R2 Score"]
)

print("\nModel Performance Comparison:")
print(results_df.sort_values(by="RMSE"))
