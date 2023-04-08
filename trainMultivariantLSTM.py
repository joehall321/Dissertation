import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras import optimizers
from DataFormatter import DataFromatter
import joblib

def NormaliseData(trial_data):
    for idx in range(len(trial_data)):
        trial = trial_data[idx]
        trial_data[idx] = pd.DataFrame(data=scaler.transform(trial))

train_path = "Datasets/force_pose/train.json"
val_path   = "Datasets/force_pose/val.json"
model_name = "model1_squatjump"

data_formatter = DataFromatter()

# Train data
train_data = data_formatter.loadData(train_path)
train_data_movements = data_formatter.formatTrialMovements(train_data)

# Validation data
val_data = data_formatter.loadData(val_path)
val_data_movements = data_formatter.formatTrialMovements(val_data)

movements = list(train_data_movements.keys())
movements.sort()
print()
print("List of movements in trial data:")
print(movements)
print()

# Filter and return list of trials that are doing a specified movement
train_trials = data_formatter.getMovementTrials("squat_jump", train_data_movements)
val_trials = data_formatter.getMovementTrials("squat_jump", val_data_movements)

# Trial data has shape (number of force data points, (17*3d keypoints + 2*3d forces + subject mass)
# 3D keypoints are expanded by incremental difference as force captured @600hz compared to keypoints @50hz
# This needs to be changed as we are increasing noise in HPE. 
print()
print("COLLECTING & FORMATTING TRAIN AND VALIDATION DATA")
print()
train_trial_data, df_for_scaler1 = data_formatter.collectFormatData(train_trials)
val_trial_data, df_for_scaler2 = data_formatter.collectFormatData(val_trials)

# LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
scaler = StandardScaler()
scaler = scaler.fit(pd.concat([df_for_scaler1,df_for_scaler2]))

NormaliseData(train_trial_data)
NormaliseData(val_trial_data)

# Save scaler
print("SAVING DATA SCALER")
joblib.dump(scaler, "scaler_"+model_name+".gz")

# print("Unnormalized:")
# print(df_for_scaler)
# print("Normalized trial 1 for training:")
# print(trial_data[0])

#Reformat input data into a shape: (n_samples x timesteps x n_features)
# Forming each x and y by sliding window of n_past data points   
# Each trail is seperated in training data by sliding window
trainX, trainY = data_formatter.formatDataSamples(train_trial_data)
valX, valY = data_formatter.formatDataSamples(val_trial_data)

# Data format: (17*3d keypoints + 2*3d forces + subject mass)
print('Total train data shape == {}.'.format(df_for_scaler1.shape))
# 3D data frame which has shape (Number of samples, n_past, features)
print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))
print()
print('Total validation data shape == {}.'.format(df_for_scaler2.shape))
print('valX shape == {}.'.format(valX.shape))
print('valY shape == {}.'.format(valY.shape))
print()

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# define the LSTM model
model = Sequential()
model.add(LSTM(550, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(300, activation='relu', return_sequences=False))
model.add(Dropout(0.7))
model.add(Dense(trainY.shape[1]))

# Possible exploding gradient problem so changing my optimizer
optimizer = optimizers.Adam(clipvalue=0.5)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# model.compile(optimizer='adam', loss='mse')
model.summary()

print()
print("STARTING TRAINING")
# fit the model
history = model.fit(trainX, trainY, epochs=200, batch_size=64,validation_data=(valX, valY), verbose=1)

model.save(model_name)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()