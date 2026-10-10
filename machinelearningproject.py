import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create dummy data for 200 houses
np.random.seed(42)
n_samples = 200

sqft = np.random.randint(600, 3500, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
age = np.random.randint(1, 30, n_samples)
price = (sqft * 150) + (bedrooms * 10000) - (age * 1200) + np.random.normal(0, 15000, n_samples)

df = pd.DataFrame({
    'SquareFeet': sqft,
    'Bedrooms': bedrooms,
    'Age': age,
    'Price': price
})
# Features (Inputs used to make predictions)
X = df[['SquareFeet', 'Bedrooms', 'Age']]

# Target (The actual value we want to predict)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)

# View what the model learned (weights and bias)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: ${rmse:,.2f}")
print(f"R2 Score: {r2:.4f}")
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('Actual vs Predicted House Prices')
plt.show()
new_house = pd.DataFrame({
    'SquareFeet': [2200],
    'Bedrooms': [3],
    'Age': [5]
})

predicted_price = model.predict(new_house)
print(f"Estimated Price: ${predicted_price[0]:,.2f}")