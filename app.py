
import gradio as gr
import joblib
import pandas as pd
import numpy as np
import io
import base64

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
    title="Annual Energy Consumption Prediction United States",
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
    title="Energy Consumption Prediction United States",
    description="Predict total energy consumption using the Random Forest model."
)

# Launch the dashboard
# iface.launch()
# annual_iface = gr.Interface(
#     fn=predict_annual_energy_consumption,
#     inputs=annual_inputs,
#     outputs="number",
#     title="Annual Energy Consumption Prediction",
#     description="Predict total annual energy consumption using the Random Forest model."
# )
# annual_iface.launch()

# Load the actual data from CSV
import plotly.graph_objects as go

try:
    actual_data = pd.read_csv('EnergyConsumption.csv')
    actual_data = actual_data.rename(columns={'Month_numerical': 'Month'})  
    actual_data = actual_data[['Year', 'Month', 'Total Energy consumption']]  
    actual_data = actual_data.rename(columns={'Total Energy consumption': 'Energy Consumption'})  
except FileNotFoundError:
    actual_data = pd.DataFrame({'Year': [], 'Month': [], 'Energy Consumption': []})
    print("EnergyConsumption.csv not found. Using empty DataFrame.")
except Exception as e:
    actual_data = pd.DataFrame({'Year': [], 'Month': [], 'Energy Consumption': []})
    print(f"Error reading EnergyConsumption.csv: {e}. Using empty DataFrame.")

def predict_and_compare(Year, Month, US_Population):
    # Convert Month from index (0-11) to actual month number (1-12)
    Month = Month + 1
    # Predict energy consumption
    predicted_consumption = predict_energy_consumption(Year, Month, US_Population)

    # Find actual consumption if available
    actual_consumption = 0
    if not actual_data.empty:
        try:
            actual_row = actual_data[(actual_data['Year'] == Year) & (actual_data['Month'] == Month)]
            if not actual_row.empty:
                actual_consumption = float(actual_row['Energy Consumption'].values[0])
        except Exception as e:
            print(f"Error finding actual data: {e}")
            actual_consumption = 0

    # Create comparison plot
    months = list(range(1, 13))
    predicted_values = [predict_energy_consumption(Year, m, US_Population) for m in months]
    actual_values = []
    
    for m in months:
        try:
            actual = actual_data[(actual_data['Year'] == Year) & (actual_data['Month'] == m)]['Energy Consumption'].values[0]
            actual_values.append(actual)
        except:
            actual_values.append(None)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=predicted_values, name='Predicted', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=months, y=actual_values, name='Actual', line=dict(color='red')))
    
    fig.update_layout(
        title=f'Energy Consumption Comparison for Year {Year}',
        xaxis_title='Month',
        yaxis_title='Energy Consumption (Quadrillion Btu)',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    difference = ((predicted_consumption - actual_consumption) / actual_consumption) * 100 if actual_consumption != 0 else 0
    return float(predicted_consumption), float(actual_consumption), float(difference), fig

comparison_inputs = [
    gr.Number(label="Year", precision=0),
    gr.Dropdown(choices=["January", "February", "March", "April", "May", "June", 
                        "July", "August", "September", "October", "November", "December"], 
                value=1, label="Month", type="index"),  
    gr.Number(label="US Population (millions)")
]

comparison_outputs = [
    gr.Number(label="Predicted Energy Consumption (Quadrillion Btu)", precision=6),
    gr.Number(label="Actual Energy Consumption (Quadrillion Btu)", precision=6),
    gr.Number(label="Error Percentage", precision=6),
    gr.Plot(label="Comparison Graph")
]

comparison_iface = gr.Interface(
    fn=predict_and_compare,
    inputs=comparison_inputs,
    outputs=comparison_outputs,
    title="Total Primary Energy Consumption United States 🗽",
    description="Predict Total Primary Energy Consumption Using Random Forest Model",
    flagging_mode="never"
)

comparison_iface.launch(show_error=True)
