
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Sample data: house sizes in square feet
X = np.array([
    [800], [1000], [1200], [1500],
    [1800], [2000], [2200], [2500]
])

# Corresponding prices in lakhs
y = np.array([
    25, 30, 35, 42, 50, 55, 60, 68
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "house_price_model.pkl")
print("Model saved successfully!")

# Predict prices for test data
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)

print("Model training completed!")
print("Actual prices:", y_test)
print("Predicted prices:", np.round(predictions, 2))
print("Mean Absolute Error:", round(mae, 2), "lakhs")

# Predict the price of a 1600 sq. ft. house
new_house = np.array([[1600]])
predicted_price = model.predict(new_house)[0]

print(
    "Predicted price for a 1600 sq. ft. house:",
    round(predicted_price, 2),
    "lakhs"
)
