import pandas as pd
import numpy as np


# 1. Load Dataset

data = pd.read_csv("stores_sales.csv", encoding="latin1")


# 2. Date Feature Engineering

data["Order Date"] = pd.to_datetime(data["Order Date"], errors="coerce")

data["Year"] = data["Order Date"].dt.year
data["Month"] = data["Order Date"].dt.month
data["Day"] = data["Order Date"].dt.day
data["Weekday"] = data["Order Date"].dt.weekday


# 3. Drop Irrelevant Columns

data.drop(
    columns=[
        "Row ID",
        "Order ID",
        "Customer Name",
        "Product Name",
        "Ship Date",
        "Order Date"
    ],
    inplace=True,
    errors="ignore"
)


# 4. Encode Categorical Features (Manual)

for col in data.select_dtypes(include="object").columns:
    data[col] = data[col].astype("category").cat.codes


# 5. Prepare X and y

X = data.drop("Sales", axis=1).values
y = data["Sales"].values.reshape(-1, 1)


# 6. Normalize Features

X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)


# 7. Train-Test Split (80-20)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# 8. Metric Functions

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r2(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

# 9. Print Prediction Functions

def print_predictions(y_true, y_pred, model_name, n=10):
    print(f"\n📊 Predictions using {model_name}")
    print("-" * 45)
    print("Actual Sales\tPredicted Sales")
    print("-" * 45)

    for i in range(min(n, len(y_true))):
        actual = y_true[i][0] if y_true.ndim > 1 else y_true[i]
        predicted = y_pred[i][0] if y_pred.ndim > 1 else y_pred[i]
        print(f"{actual:.2f}\t\t{predicted:.2f}")


# 10. Linear / Ridge / Lasso (GD)

def train_linear(X, y, lr=0.01, epochs=1000, l1=0.0, l2=0.0):
    w = np.zeros((X.shape[1], 1))
    b = 0.0

    for _ in range(epochs):
        y_pred = X @ w + b
        dw = (X.T @ (y_pred - y)) / len(X)
        db = np.mean(y_pred - y)

        # Regularization
        dw += l2 * w + l1 * np.sign(w)

        w -= lr * dw
        b -= lr * db

    return w, b

def predict_linear(X, w, b):
    return X @ w + b


# 11. kNN Regression (Manual)

def knn_predict(X_train, y_train, X_test, k=5):
    preds = []
    for x in X_test:
        distances = np.sqrt(np.sum((X_train - x) ** 2, axis=1))
        idx = np.argsort(distances)[:k]
        preds.append(np.mean(y_train[idx]))
    return np.array(preds).reshape(-1, 1)


# 12. Train Models

results = []

# Linear Regression
w_lr, b_lr = train_linear(X_train, y_train)
pred_lr = predict_linear(X_test, w_lr, b_lr)
print_predictions(y_test, pred_lr, "Linear Regression")
results.append(["Linear Regression", mae(y_test, pred_lr), rmse(y_test, pred_lr), r2(y_test, pred_lr)])

# Ridge Regression
w_ridge, b_ridge = train_linear(X_train, y_train, l2=0.1)
pred_ridge = predict_linear(X_test, w_ridge, b_ridge)
print_predictions(y_test, pred_ridge, "Ridge Regression")
results.append(["Ridge Regression", mae(y_test, pred_ridge), rmse(y_test, pred_ridge), r2(y_test, pred_ridge)])

# Lasso Regression
w_lasso, b_lasso = train_linear(X_train, y_train, l1=0.05)
pred_lasso = predict_linear(X_test, w_lasso, b_lasso)
print_predictions(y_test, pred_lasso, "Lasso Regression")
results.append(["Lasso Regression", mae(y_test, pred_lasso), rmse(y_test, pred_lasso), r2(y_test, pred_lasso)])

# kNN Regression
pred_knn = knn_predict(X_train, y_train, X_test, k=7)
print_predictions(y_test, pred_knn, "kNN Regression")
results.append(["kNN Regression", mae(y_test, pred_knn), rmse(y_test, pred_knn), r2(y_test, pred_knn)])


# 12. Comparison Table

comparison = pd.DataFrame(
    results,
    columns=["Model", "MAE", "RMSE", "R2 Score"]
)

print("\n📌 MODEL COMPARISON TABLE")
print(comparison.sort_values(by="RMSE"))