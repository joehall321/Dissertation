import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras import optimizers
from DataFormatter import DataFromatter

train_path = "Datasets/force_pose/train.json"
val_path   = "Datasets/force_pose/val.json"

    
movement = "all"
sample_size = 30
epochs   = 50

parameters = "ss"+str(sample_size)+"_blstm_64_1024_dense256_lr1e-4_bs64_eps"+str(epochs)
model_name = movement+"-"+parameters

data_formatter = DataFromatter()

# Train data
train_data = data_formatter.loadData(train_path)
train_data_movements = data_formatter.formatTrialMovements(train_data)

# Validation data
val_data = data_formatter.loadData(val_path)
val_data_movements = data_formatter.formatTrialMovements(val_data, flip_xz_flag=True)

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
train_trial_data = data_formatter.collectFormatData(train_trials,"linearinterpolation" in model_name)
val_trial_data   = data_formatter.collectFormatData(val_trials,"linearinterpolation" in model_name)

# LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
train_trial_data = data_formatter.NormaliseData(train_trial_data, model_name)
val_trial_data = data_formatter.NormaliseData(val_trial_data, model_name)

#Reformat input data into a shape: (n_samples x timesteps x n_features)
# Forming each x and y by sliding window of n_past data points   
# Each trail is seperated in training data by sliding window
trainX, trainY, _ = data_formatter.formatDataSamples(train_trial_data, sample_size, "endframeforce" in model_name)
valX, valY, _ = data_formatter.formatDataSamples(val_trial_data, sample_size, "endframeforce" in model_name)

# Data format: (17*3d keypoints + 2*3d forces + subject mass)
# 3D data frame which has shape (Number of samples, n_past, features)
print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))
print()
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
plt.suptitle(model_name+" Training")
plt.plot(history.history['loss'], label='MSE Training loss')
plt.legend()
plt.savefig("Models/"+model_name+"/training_loss.png")
plt.plot(history.history['val_loss'], label='MSE Validation loss')
plt.legend()
plt.savefig("Models/"+model_name+"/training_loss_val.png")
plt.close()