import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras import optimizers
from DataFormatter import DataFromatter
import joblib

test_path = "Datasets/force_pose/val.json"
model_name = "model1_squatjump"

data_formatter = DataFromatter()

data = data_formatter.loadData(test_path)
data_movements = data_formatter.formatTrialMovements(data)

movements = list(data_movements.keys())
movements.sort()
print(movements)
print()

# Get list of trials that are doing a specified movement
test_trials = data_formatter.getMovementTrials("squat_jump", data_movements)

# Trial data has shape (number of force data points, (17*3d keypoints + 2*3d forces + subject mass)
# 3D keypoints are expanded by incremental difference as force captured @600hz compared to keypoints @50hz
# This needs to be changed as we are increasing noise in HPE. 
trial_data, _ = data_formatter.collectFormatData(test_trials)

print("Before normalization trial data:")
print(trial_data[0].iloc[59:69])
# Values need to be normalized using smae scaler when training
# normalize the dataset
scaler = joblib.load("scaler_"+model_name+".gz")
for idx in range(len(trial_data)):
    trial = trial_data[idx]
    trial_data[idx] = pd.DataFrame(data=scaler.transform(trial))

# print("Unnormalized:")
# print(df_for_scaler)
# print("Normalized trial 1 for training:")
# print(trial_data[0])

#Reformat input data into a shape: (n_samples x timesteps x n_features)
# Forming each x and y by sliding window of n_past data points   
# Each trail is seperated in training data by sliding window
testX, normalized_testY = data_formatter.formatDataSamples([trial_data[0]])
# Load model
model = load_model(model_name)

#Make prediction
normalized_prediction = model.predict(testX)
print()
print("Shape of val data:",testX.shape)
print("Shape of prediction data:",normalized_prediction.shape)
# print("Normalized Prediction:",normalized_prediction)
# print("Normalized Truth:",testY)

#Perform inverse transformation to rescale back to original range
#Since we used 5 variables for transform, the inverse expects same dimensions
#Therefore, let us copy our values 5 times and discard them after inverse transform
prediction = pd.DataFrame(index=range(normalized_prediction.shape[0]),columns=range(trial.shape[1]))
truth = pd.DataFrame(index=range(normalized_prediction.shape[0]),columns=range(trial.shape[1]))
# Replace force columns in copy with normalized model prediction columns
prediction.loc[:, 52:57] = normalized_prediction
truth.loc[:, 52:57] = normalized_testY

prediction = pd.DataFrame(scaler.inverse_transform(prediction)).loc[:,52:57]
truth = pd.DataFrame(scaler.inverse_transform(truth)).loc[:,52:57]

print("Truth:")
print(truth)
print()
print("Prediction")
print(prediction)
print()