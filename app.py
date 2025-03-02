
import gradio as gr
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('model/random_forest_model.pkl')

# Define the input components
inputs = [
    gr.Number(label="Year", precision=0),  # Year input (integer)
    gr.Dropdown(choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], label="Month"), # Month input (integer)
    gr.Number(label="US Population (millions)")
]

# Define the input components for annual prediction
annual_inputs = [
    gr.Number(label="Year", precision=0),  # Year input (integer)
    gr.Number(label="US Population (millions)")
]

# Define the annual prediction function
def predict_annual_energy_consumption(Year, US_Population):
    annual_consumption = sum(predict_energy_consumption(Year, month, US_Population) for month in range(1, 13))
    return annual_consumption

# Create the Gradio interface for annual prediction
annual_iface = gr.Interface(
    fn=predict_annual_energy_consumption,
    inputs=annual_inputs,
    outputs="number",
    title="Annual Energy Consumption Prediction",
    description="Predict total annual energy consumption using the Random Forest model."
)
# Define the prediction function
def predict_energy_consumption(Year, Month, US_Population):
    # Feature engineering for Year and Month
    Year_scaled = 2 * np.pi * (Year - 1973) / (2024 - 1973)  # Scale Year
    Month_scaled = 2 * np.pi * (Month - 1) / (12 - 1)       # Scale Month

    Year_sin = np.sin(Year_scaled)
    Year_cos = np.cos(Year_scaled)
    Month_sin = np.sin(Month_scaled)
    Month_cos = np.cos(Month_scaled)
    
    # Interaction and polynomial features
    Population_Year_sin = US_Population * Year_sin
    Population_Year_cos = US_Population * Year_cos
    Population_Month_sin = US_Population * Month_sin
    Population_Month_cos = US_Population * Month_cos
    Population_squared = US_Population ** 2

    # Create input DataFrame
    input_data = pd.DataFrame([[US_Population, Year_sin, Year_cos, Month_sin, Month_cos, Population_Year_sin, Population_Year_cos, Population_Month_sin, Population_Month_cos, Population_squared]], 
                             columns=['US Population (millions)', 'Year_sin', 'Year_cos', 'Month_sin', 'Month_cos', 'Population_Year_sin', 'Population_Year_cos', 'Population_Month_sin', 'Population_Month_cos', 'Population_squared'])
    
    prediction = model.predict(input_data)[0]
    return prediction

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_energy_consumption,
    inputs=inputs,
    outputs="number",
    title="Energy Consumption Prediction",
    description="Predict total energy consumption using the Random Forest model."
)

# Launch the dashboard
# iface.launch()
annual_iface = gr.Interface(
    fn=predict_annual_energy_consumption,
    inputs=annual_inputs,
    outputs="number",
    title="Annual Energy Consumption Prediction",
    description="Predict total annual energy consumption using the Random Forest model."
)
annual_iface.launch()