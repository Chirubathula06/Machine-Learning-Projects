# Stock Price Prediction using LSTM (FINAL VERSION)

# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input

# Step 2: Sample Stock Data
data = [100, 102, 105, 107, 110, 115, 120, 125, 130, 128,
        135, 140, 145, 150, 155, 160, 165, 170, 175, 180]

df = pd.DataFrame(data, columns=["Price"])

# Step 3: Normalize Data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Step 4: Create Sequences
X = []
y = []
time_step = 3

for i in range(len(scaled_data) - time_step):
    X.append(scaled_data[i:i+time_step])
    y.append(scaled_data[i+time_step])

X, y = np.array(X), np.array(y)

# Step 5: Train-Test Split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Step 6: Build LSTM Model (NO WARNING)
model = Sequential([
    Input(shape=(time_step, 1)),
    LSTM(50, return_sequences=True),
    LSTM(50),
    Dense(1)
])

# Step 7: Compile Model
model.compile(optimizer='adam', loss='mse')

# Step 8: Train Model
history = model.fit(X_train, y_train, epochs=100, verbose=0)

# Step 9: Predict
predictions = model.predict(X_test, verbose=0)

# Convert back to original values
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test)

# =========================
# 📊 PRINT OUTPUT FIRST
# =========================

print("\n==============================")
print("📊 LSTM STOCK PREDICTION RESULT")
print("==============================")

print("\nActual Prices:")
print(y_test_actual.flatten())

print("\nPredicted Prices:")
print(predictions.flatten())

# Predict next day price
last_sequence = scaled_data[-time_step:]
last_sequence = last_sequence.reshape(1, time_step, 1)

next_price = model.predict(last_sequence, verbose=0)
next_price = scaler.inverse_transform(next_price)

print("\n📈 Predicted Next Day Price:", next_price[0][0])

# =========================
# 📉 GRAPH VISUALIZATION
# =========================

plt.figure()
plt.plot(y_test_actual, label="Actual Price")
plt.plot(predictions, label="Predicted Price")
plt.title("Stock Price Prediction (LSTM)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
