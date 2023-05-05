import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras import optimizers
from DataFormatter import DataFromatter
import joblib

def NormaliseData(trial_data):
    for idx in range(len(trial_data)):
        trial = trial_data[idx]
        trial_data[idx] = pd.DataFrame(data=scaler.transform(trial))

train_path = "Datasets/force_pose/train.json"
val_path   = "Datasets/force_pose/val.json"

movement   = "all"
sample_size = 12
epochs = 20

parameters = "ss"+str(sample_size)+"_blstm_64_1024_dense256_lr1e-4_bs64_eps"+str(epochs)
model_name = movement+"-"+parameters

data_formatter = DataFromatter()

# Train data
train_data = data_formatter.loadData(train_path)
train_data_movements = data_formatter.formatTrialMovements(train_data)

# Validation data
val_data = data_formatter.loadData(val_path)
val_data_movements = data_formatter.formatTrialMovements(val_data)

print()
print("List of movements in training data:")
print(data_formatter.dict_dims(train_data_movements))
print()
print("List of movements in validation data:")
print(data_formatter.dict_dims(val_data_movements))
print()

# Filter and return list of trials that are doing a specified movement
train_trials = data_formatter.getMovementTrials(movement, train_data_movements)
val_trials = data_formatter.getMovementTrials(movement, val_data_movements)

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
joblib.dump(scaler, "Models/scalers/scaler_"+model_name+".gz")

# print("Unnormalized:")
# print(df_for_scaler)
# print("Normalized trial 1 for training:")
# print(trial_data[0])

#Reformat input data into a shape: (n_samples x timesteps x n_features)
# Forming each x and y by sliding window of n_past data points   
# Each trail is seperated in training data by sliding window
trainX, trainY = data_formatter.formatDataSamples(train_trial_data, sample_size)
valX, valY = data_formatter.formatDataSamples(val_trial_data, sample_size)

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

model.add(Bidirectional(LSTM(64, activation='relu', return_sequences=True), input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Bidirectional(LSTM(1024, activation='relu', return_sequences=False)))
model.add(Dense(256))
model.add(Dense(trainY.shape[1]))

# model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(trainX.shape[1], trainX.shape[2])))
# model.add(LSTM(1024, activation='relu', return_sequences=False))
# model.add(Dense(256))
# model.add(Dense(trainY.shape[1]))

# Possible exploding gradient problem so changing my optimizer
optimizer = optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])

# model.compile(optimizer='adam', loss='mse')
model.build(input_shape=(trainX.shape[1], trainX.shape[2]))
model.summary()

print()
print("STARTING TRAINING")
# fit the model 
# validation_data=(valX, valY)
history = model.fit(trainX, trainY, epochs=epochs, batch_size=64, validation_data=(valX, valY), verbose=1)

model.save("Models/"+model_name)

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(valX, valY, batch_size=64)
print("Loss, RMSE:", results)

# Attaching 2D axis to the figure
fig, axis = plt.subplots(1, 2)
axis[0].plot(history.history['loss'], label='Training loss')
axis[0].plot(history.history['val_loss'], label='Validation loss')
axis[1].plot(history.history['root_mean_squared_error'], label='RMSE')
axis[1].plot(history.history['val_root_mean_squared_error'], label='Validation RMSE')
axis[0].title.set_text('Loss history')
axis[1].title.set_text('RMSE history')
axis[0].legend()
axis[1].legend()
plt.show()