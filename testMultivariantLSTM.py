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

def getSampleSize(model_name):
    idx = model_name.find("ss")+2
    size = ""
    while model_name[idx]!='_':
        size+=model_name[idx]
        idx+=1
    return int(size)


test_path = "Datasets/force_pose/val.json"
model_name = "all-ss12_lstm_64_1024_dense256_lr1e-4_bs64_eps100"

sample_size = getSampleSize(model_name)

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
scaler = joblib.load("Models/scalers/scaler_"+model_name+".gz")
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
print("Sample size:",sample_size)
testX, normalized_testY = data_formatter.formatDataSamples([trial_data[0]],sample_size)
# Load model
model = load_model("Models/"+model_name)

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
prediction.loc[:, 51:56] = normalized_prediction
truth.loc[:, 51:56] = normalized_testY

prediction = pd.DataFrame(scaler.inverse_transform(prediction)).loc[:,51:56]
truth = pd.DataFrame(scaler.inverse_transform(truth)).loc[:,51:56]

# # Negatate values of xs and zs
# prediction[51] = -prediction[51]
# prediction[53] = -prediction[53]
# prediction[54] = -prediction[54]
# prediction[56] = -prediction[56]

print("Truth:")
print(truth)
print()
print("Prediction")
print(prediction)
print()

# Attaching 2D axis to the figure
fig, axis = plt.subplots(2, 3)
axis[0,0].title.set_text('ground_force1_vx')
axis[0,0].plot(truth[51], label='Truth')
axis[0,0].plot(-prediction[51], label='LSTM prediction')
axis[0,0].legend()

axis[0,1].title.set_text('ground_force1_vy')
axis[0,1].plot(truth[52], label='Truth')
axis[0,1].plot(prediction[52], label='LSTM prediction')
axis[0,1].legend()

axis[0,2].title.set_text('ground_force1_vz')
axis[0,2].plot(truth[53], label='Truth')
axis[0,2].plot(-prediction[53], label='LSTM prediction')
axis[0,2].legend()

axis[1,0].title.set_text('ground_force2_vx')
axis[1,0].plot(truth[54], label='Truth')
axis[1,0].plot(-prediction[54], label='LSTM prediction')
axis[1,0].legend()

axis[1,1].title.set_text('ground_force2_vy')
axis[1,1].plot(truth[55], label='Truth')
axis[1,1].plot(prediction[55], label='LSTM prediction')
axis[1,1].legend()

axis[1,2].title.set_text('ground_force2_vz')
axis[1,2].plot(truth[56], label='Truth')
axis[1,2].plot(-prediction[56], label='LSTM prediction')
axis[1,2].legend()

plt.show()