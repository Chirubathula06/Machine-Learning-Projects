# Car Price Prediction using ANN (FINAL FIXED)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Step 1: Dataset
data = {
    "Engine_Size": [1000, 1200, 1500, 1800, 2000, 2200, 2500, 3000],
    "Mileage": [20, 18, 15, 12, 10, 9, 8, 7],
    "Age": [5, 4, 3, 3, 2, 2, 1, 1],
    "Price": [300000, 350000, 500000, 650000, 700000, 800000, 900000, 1200000]
}

df = pd.DataFrame(data)

# Step 2: Features & Target
X = df[["Engine_Size", "Mileage", "Age"]]
y = df["Price"]

# Step 3: Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Scale X and y
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test = scaler_y.transform(y_test.values.reshape(-1, 1))

# Step 5: Build ANN (FIXED)
model = Sequential([
    Input(shape=(3,)),   # ✅ correct way
    Dense(8, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1)
])

# Step 6: Compile
model.compile(optimizer='adam', loss='mse')

# Step 7: Train
model.fit(X_train, y_train, epochs=200, verbose=0)

# Step 8: Predict
y_pred = model.predict(X_test)

# Convert back to original scale
y_pred_actual = scaler_y.inverse_transform(y_pred)
y_test_actual = scaler_y.inverse_transform(y_test)

# Step 9: Evaluate
mse = mean_squared_error(y_test_actual, y_pred_actual)
print("Mean Squared Error:", mse)

# Step 10: Predict New Car
new_car = pd.DataFrame({
    "Engine_Size": [2000],
    "Mileage": [12],
    "Age": [2]
})

new_car_scaled = scaler_X.transform(new_car)
pred_scaled = model.predict(new_car_scaled)

# Convert back to real price
predicted_price = scaler_y.inverse_transform(pred_scaled)

print("\nPredicted Car Price:", predicted_price[0][0])
