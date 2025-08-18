# Import libraries
from keras.models import Sequential, load_model
import numpy as np
import pandas as pd
import pickle
import os

# Get current directory
current_directory = os.getcwd()

# Construct the file path for the scaler
scaler_X_path = os.path.join(current_directory, 'scaler_X.pkl')
scaler_y_path = os.path.join(current_directory, 'scaler_y.pkl')

# Load the saved scaler
with open(scaler_X_path, 'rb') as f:
    scaler_X = pickle.load(f)

with open(scaler_y_path, 'rb') as f:
    scaler_y = pickle.load(f)

# Construct the file path for the Input_data.xlsx
file_path = os.path.join(current_directory, 'Input_data.xlsx')

# Load the data
data = pd.read_excel(file_path, dtype=float)
data_array= np.array(data)
X = data_array[:, [0, 1, 2, 3, 4, 5, 6]]

# Standardize the input data using z-score normalization
X_normalized = scaler_X.transform(X)

# load model
model = load_model("基本model.h5")
#model = load_model("基本model.h5")

# prediction
y_predict = model.predict(X_normalized)

# Denormalize data
y_predict_denormalized = scaler_y.inverse_transform(y_predict)

# Exponentiate the prediction results
y = np.exp(y_predict_denormalized)

# Convert the prediction results into a DataFrame and name the columns
output_data = pd.DataFrame(y, columns=[
    'PGA', 'PGV', 'Sa0.01s', '0.02', '0.03', '0.05', '0.075', '0.1', '0.15', '0.2',
    '0.25', '0.3', '0.4', '0.5', '0.75', '1', '1.5', '2', '2.5', '3', '3.5', '4',
    '5', '7.5', '10'
])

# Save to an Excel file
output_data.to_excel('Output_data.xlsx', index=False)