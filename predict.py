
import joblib
import numpy as np

# Load the saved model
model = joblib.load("house_price_model.pkl")

# Enter the house size in square feet
size = float(input("Enter house size in square feet: "))

# Predict the house price
house = np.array([[size]])
price = model.predict(house)[0]

print(f"Predicted house price: {price:.2f} lakhs")
