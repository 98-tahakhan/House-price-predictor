from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('data.csv')

# 0) Drop zero-price or obviously invalid rows
df = df[df['price'] > 0].copy()

# 1) Basic stats
print("Rows after removing zero-price:", len(df))
print(df['price'].describe())

# 2) Remove top 1% most expensive outliers (you can change to 0.995 etc)
upper = df['price'].quantile(0.99)
df = df[df['price'] <= upper].copy()
print("Rows after trimming top 1%:", len(df), "price max:", df['price'].max())

# 3) Date -> year (if date present)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year.fillna(df['date'].dt.year.mode()[0]).astype(int)

# 4) Select features (include extras if available)
candidate_features = [
    'sqft_living', 'bedrooms', 'bathrooms', 'sqft_above', 'sqft_basement',
    'floors', 'waterfront', 'view', 'condition', 'yr_built', 'yr_renovated',
    'lat','long','year'
]
features = [f for f in candidate_features if f in df.columns]

# 5) City encoding: keep top N frequent cities
if 'city' in df.columns:
    top_n = 10
    top_cities = df['city'].value_counts().nlargest(top_n).index
    df['city_top'] = df['city'].where(df['city'].isin(top_cities), other='Other')
    features.append('city_top')

X = df[features].copy()
y = df['price'].copy()

# 6) Fill missing numeric values with median
for col in X.select_dtypes(include=[np.number]).columns:
    X[col].fillna(X[col].median(), inplace=True)

# 7) One-hot encode city_top (if used)
if 'city_top' in X.columns:
    X = pd.get_dummies(X, columns=['city_top'], drop_first=True)

# 8) Log-transform the target
y_log = np.log1p(y)

# 9) Split
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

# 10) Train Linear Regression (log-space)
lr = LinearRegression()
lr.fit(X_train, y_train_log)
y_pred_log_lr = lr.predict(X_test)

# 11) Train RandomForest (log-space)
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train_log)
y_pred_log_rf = rf.predict(X_test)

# 12) Helper to evaluate and print both log-space and original-space metrics
def eval_and_print(y_test_log, y_pred_log, name):
    # log-space metrics
    mse_log = mean_squared_error(y_test_log, y_pred_log)
    r2_log = r2_score(y_test_log, y_pred_log)
    print(f"\n{name} - Log-space -> MSE: {mse_log:.4f}  R2: {r2_log:.4f}")
    # back to original space
    y_test = np.expm1(y_test_log)
    y_pred = np.expm1(y_pred_log)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    # MAPE (avoid div by zero)
    mask = y_test != 0
    mape = (np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])).mean() * 100
    r2_orig = r2_score(y_test, y_pred)
    print(f"{name} - Original-space -> RMSE: {rmse:,.2f}  MAE: {mae:,.2f}  MAPE: {mape:.2f}%  R2: {r2_orig:.4f}")

eval_and_print(y_test_log, y_pred_log_lr, "LinearRegression")
eval_and_print(y_test_log, y_pred_log_rf, "RandomForest")

# predictions back to original space
y_test = np.expm1(y_test_log)
y_pred_lr = np.expm1(y_pred_log_lr)

residuals = y_test - y_pred_lr

plt.figure(figsize=(8,5))
plt.scatter(y_pred_lr, residuals, alpha=0.4)
plt.hlines(0, xmin=y_pred_lr.min(), xmax=y_pred_lr.max(), colors='r')
plt.xlabel("Predicted Price")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residuals vs Predicted (LinearRegression)")
plt.show()

# Save LinearRegression and RandomForest
joblib.dump(lr, 'linear_reg_log_target.pkl')
joblib.dump(rf, 'random_forest_log_target.pkl')

# Save feature columns for future use
feature_cols = X.columns.tolist()
joblib.dump(feature_cols, 'model_feature_cols.pkl')

print("Saved lr, rf, and feature_cols.")