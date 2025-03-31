# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Load Sierra Leone dataset (replace with your actual dataset)
data = {
    "MedianIncome": [5000, 6000, 7000, 8000, 9000],
    "PopulationDensity": [100, 200, 300, 400, 500],
    "RoomsPerHouse": [3, 4, 3, 5, 4],
    "MedHouseValue": [100000, 120000, 150000, 180000, 200000]
}
df = pd.DataFrame(data)

# Separate features (X) and target variable (y)
X = df.drop('MedHouseValue', axis=1)  # Features
y = df['MedHouseValue']  # Target

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Train a Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Save the scaler and models
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(lr_model, "models/linear_regression_model.joblib")
joblib.dump(dt_model, "models/decision_tree_model.joblib")

print("Models and scaler saved successfully!")