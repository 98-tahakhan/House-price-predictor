House Price Predictor — Machine Learning Project

This project builds a House Price Prediction model using Linear Regression and Random Forest Regressor on real estate data.
The model predicts house prices based on key property features like:

Square footage

Bedrooms

Bathrooms

Floors

Basement size

Waterfront, view, condition

Construction year

City (encoded)

Price (log-transformed target)

This project was completed as part of my Hands-On Machine Learning practice (Day-10).

📁 Project Structure
House-Price-Predictor/
│
├── data.csv                         # Dataset
├── House-Price-Predictor.py         # Main ML pipeline
├── linear_reg_log_target.pkl        # Saved Linear Regression model
├── random_forest_log_target.pkl     # Saved Random Forest model
├── model_feature_cols.pkl           # Feature columns used during training
├── README.md                        # Documentation
└── .gitignore

🚀 Features
✔️ Cleans & preprocesses real-world house data

Drops zero-price anomalies

Removes top 1% outliers

Extracts year from date

Encodes top 10 cities using One-Hot Encoding

Applies log-transform to stabilize price distribution

✔️ Trains two models

Linear Regression (log-space)

Random Forest Regressor (log-space)

✔️ Evaluation metrics printed for both models

RMSE

MAE

MAPE

R² Score

✔️ Saved models included (.pkl)

Can be loaded anytime for prediction.

📊 Results (Current Model)
Model	RMSE	MAE	MAPE	R² Score
Linear Regression	~167,556	~107,876	~21.37%	0.6396
Random Forest	~177,781	~115,782	~23.46%	0.5943

📚 Technologies Used
Python

NumPy

Pandas

Scikit-Learn

Matplotlib (optional)

📌 Future Improvements

Add advanced models (XGBoost, Gradient Boosting)

Add interactive UI using Streamlit

Add full hyperparameter tuning (GridSearch / RandomSearch)

Deploy as API using Flask or FastAPI