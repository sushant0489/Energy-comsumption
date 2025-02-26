import joblib
import pandas as pd

# Load the trained model
model = joblib.load("model/energy_consumption_model.pkl")


# Define input for prediction (Example: October 2024)
future_input = pd.DataFrame({"Year": [2024], "Month_Num": [11]})

# Predict total energy consumption
predicted_consumption = model.predict(future_input)[0]

print(f"Predicted Total Energy Consumption: {predicted_consumption}")