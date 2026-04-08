# House Price Prediction with Graph Visualization

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Create Sample Dataset
data = {
    "Area": [1000, 1500, 1800, 2400, 3000, 3500, 4000, 4500],
    "Bedrooms": [2, 3, 3, 4, 4, 5, 5, 6],
    "Bathrooms": [1, 2, 2, 3, 3, 4, 4, 5],
    "Price": [200000, 300000, 350000, 450000, 500000, 600000, 650000, 750000]
}

df = pd.DataFrame(data)

# Step 3: Define Features and Target
X = df[["Area", "Bedrooms", "Bathrooms"]]
y = df["Price"]

# Step 4: Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict
y_pred = model.predict(X_test)

# Step 7: Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Step 8: Fix warning using DataFrame
new_house = pd.DataFrame({
    "Area": [2500],
    "Bedrooms": [4],
    "Bathrooms": [3]
})

predicted_price = model.predict(new_house)
print("\nPredicted Price:", predicted_price[0])

# =========================
# 📊 GRAPH VISUALIZATION
# =========================

# 1️⃣ Actual vs Predicted Scatter Plot
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()

# 2️⃣ Area vs Price (Regression View)
plt.figure()
plt.scatter(df["Area"], df["Price"])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area vs Price")
plt.show()

# 3️⃣ Bedrooms vs Price
plt.figure()
plt.scatter(df["Bedrooms"], df["Price"])
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.title("Bedrooms vs Price")
plt.show()
