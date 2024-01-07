import streamlit as st
import joblib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init

feature_stats = joblib.load("pickles/feature_stats.pkl")

# Access the mean and standard deviation values for the pressure feature
pressure_mean = feature_stats['pressure_mean']
pressure_std = feature_stats['pressure_std']

# Access the mean and standard deviation values for the temperature feature
temperature_mean = feature_stats['temperature_mean']
temperature_std = feature_stats['temperature_std']

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Create the MLP
        torch.manual_seed(42)

        # Flatten layer
        self.flatten = nn.Flatten()

        # 1st hidden layer
        self.fc1 = nn.Linear(config.n_in, config.n_hidden)
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')

        # ReLU activation function
        self.relu = nn.ReLU()

        # Output layer
        self.fc2 = nn.Linear(config.n_hidden, config.n_out)
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, X):
        X = self.flatten(X)
        X = self.relu(self.fc1(X))
        pred_density = self.fc2(X)
        return pred_density

# Load the pickled model
model = joblib.load('pickles/model_mlp.pkl')

# Define the input data dictionary
sample_data_1 = {
    'Pressure': 30,
    'Temperature': 350,
    'Methane': 0.759522443,
    'Ethane': 0.086174251,
    'Propane': 0.0584332,
    '2-methylpropane': 0.012103352,
    'Butane': 0.01393312,
    '2-metylbutane': 0,
    'Pentane': 0.006430073,
    'Octane': 0.001287943,
    'Toluene': 0.000643971,
    'Methylcyclopentane': 0,
    'Nitrogen': 0.061251704,
    'Carbon Dioxide': 0.00000528,
}

# Define the function for making predictions
def inference_single_data(data_dict: dict):
    required_keys = set(['Pressure', 'Temperature', 'Methane', 'Ethane', 'Propane', '2-methylpropane', 'Butane', '2-metylbutane', 'Pentane', 'Octane', 'Toluene', 'Methylcyclopentane', 'Nitrogen', 'Carbon Dioxide'])
    if set(data_dict.keys()) != required_keys:
        raise Exception(f"The data sequence must follow this: {required_keys}")

    data_dict_arr = np.array(list(data_dict.values()))

    # pressure_mean, presure_std, temperature_mean, temperature_std are data calculated from train_data
    data_dict_arr = data_dict_arr.copy()
    data_dict_arr[0] = (data_dict_arr[0] - pressure_mean) / pressure_std
    data_dict_arr[1] = (data_dict_arr[1] - temperature_mean) / temperature_std

    # Convert NumPy array to PyTorch tensor and convert to float data type
    input_tensor = torch.from_numpy(data_dict_arr).float()

    # Predict the single input using the model
    y_test_pred = model(input_tensor.unsqueeze(0))

    return y_test_pred.item()

# Define the main function for the Streamlit app
def main():
    # Set the page title
    st.title('Density Prediction App')

    # Display the input form
    st.subheader('Enter Input Data')
    pressure = st.number_input('Pressure (MPa)', value=sample_data_1['Pressure'])
    temperature = st.number_input('Temperature (K)', value=sample_data_1['Temperature'])
    methane = st.number_input('Methane (mol fraction)', value=sample_data_1['Methane'])
    ethane = st.number_input('Ethane (mol fraction)', value=sample_data_1['Ethane'])
    propane = st.number_input('Propane (mol fraction)', value=sample_data_1['Propane'])
    two_methylpropane = st.number_input('2-methylpropane (mol fraction)', value=sample_data_1['2-methylpropane'])
    butane = st.number_input('Butane (mol fraction)', value=sample_data_1['Butane'])
    two_metylbutane = st.number_input('2-metylbutane (mol fraction)', value=sample_data_1['2-metylbutane'])
    pentane = st.number_input('Pentane (mol fraction)', value=sample_data_1['Pentane'])
    octane = st.number_input('Octane (mol fraction)', value=sample_data_1['Octane'])
    toluene = st.number_input('Toluene (mol fraction)', value=sample_data_1['Toluene'])
    methylcyclopentane = st.number_input('Methylcyclopentane (mol fraction)', value=sample_data_1['Methylcyclopentane'])
    nitrogen = st.number_input('Nitrogen (mol fraction)', value=sample_data_1['Nitrogen'])
    carbon_dioxide = st.number_input('Carbon Dioxide (mol fraction)', value=sample_data_1['Carbon Dioxide'])

    # Display the predicted density
    if st.button('Predict'):
        input_data = {
            'Pressure': pressure,
            'Temperature': temperature,
            'Methane': methane,
            'Ethane': ethane,
            'Propane': propane,
            '2-methylpropane': two_methylpropane,
            'Butane': butane,
            '2-metylbutane': two_metylbutane,
            'Pentane': pentane,
            'Octane': octane,
            'Toluene': toluene,
            'Methylcyclopentane': methylcyclopentane,
            'Nitrogen': nitrogen,
            'Carbon Dioxide': carbon_dioxide,
        }
        try:
            density = inference_single_data(input_data)
            st.write(f'Predicted Density: {density:.4f} kg/m3')
        except Exception as e:
            st.write(e)

# Run the main function
if __name__ == '__main__':
    main()