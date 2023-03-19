import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras import optimizers

train_path = "Datasets/force_pose/train.json"

subject_masses= {
"Subject1" : 83.25,
"Subject2" : 86.48,
"Subject3" : 87.54,
"Subject4" : 86.11,
"Subject5" : 74.91,
"Subject6" : 111.91,
"Subject7" : 82.64,
"Subject8" : 90.44 }

# Load data
def loadData(path):
    data = json.load(open(path))
    return data

# Puts categrises each trial into movement list 
def formatTrialMovements(train_data):
    train_movements = {}
    for trial in train_data:
        movement=trial.get("movement")
        train_movements.setdefault(movement,[])
        train_movements.get(movement).append(trial)
    return train_movements

# Returns joined list of trials for specified movement
def getMovementTrials(movement, data):
    movement_data = []
    for type in data:
        if movement in type.lower():
            movement_data.extend(data[type])
    return movement_data

# Return dictonary of {x1:[x1,...,xt], y1:[y1,...,yt], z1:[z1,...,zt],
#                   ..., x51:[x1,...,xt],y51:[y1,...,yt], z51:[z1,...,zt]}
def getTriangulatedPose(trial):
    frames = trial.get("frames")
    mass=subject_masses.get(trial.get("subject"))

    # Create dictionary with empty list for 51 points
    keypoints_data = {}
    keypoints_data.setdefault("mass",[])
    for num in range(1,18):
        keypoints_data.setdefault("x"+str(num),[])
        keypoints_data.setdefault("y"+str(num),[])
        keypoints_data.setdefault("z"+str(num),[])

    # Populate point lists with data from each frame in trial
    counter=0
    for frame in frames:
        pose_kps = frame.get("triangulated_pose")
        keypoints_data.get("mass").append(mass)
        for kp_idx in range(len(pose_kps)):
            x,y,z = pose_kps[kp_idx]
            keypoints_data.get("x"+str(kp_idx+1)).append(x)
            keypoints_data.get("y"+str(kp_idx+1)).append(y)
            keypoints_data.get("z"+str(kp_idx+1)).append(z)
        counter+=1
        #if(counter==2): break
    return keypoints_data

# Populates pose data so that the number of datapoints matches force data. 
# Force data captured at 600hz
# Pose data captured at 50hz
# Data point gaps are filled by using incremental difference
def populatePoseGaps(pose_data, forces_length):
    num_extra_points = 11
    # print("Extra points per x:",num_extra_points)
    # Loop through keypoints list
    for kp_label in pose_data:
        kps = pose_data.get(kp_label)
        increased_kps=[]

        #Loop through list
        for kp_idx in range(len(kps)):
            kp_value = kps[kp_idx]

            if kp_idx<len(kps)-1:
                kp_next_value= kps[kp_idx+1]
                increased_kps.append(kp_value)
                inc=(kp_next_value-kp_value)/num_extra_points
                for _ in range(num_extra_points):
                    kp_value+=inc
                    increased_kps.append(kp_value)
            else:
                while len(increased_kps)<forces_length:
                    increased_kps.append(kp_value)

        pose_data[kp_label]=increased_kps
    return pose_data
    
def addForcesToData(data, forces):
    for force_label in forces:
        if force_label!="time":
            data[force_label] = forces[force_label]
    return data

# Collects and formats data
# creates a list of dataframes 
def collectFormatData(train_trials):
    print()
    print("COLLECTING & FORMATTING DATA")
    print()
    trial_data = []

    for idx in range(len(train_trials)):
        trial = train_trials[idx]
        forces = trial.get("grf")
        trial_label = "trail_"+str(idx+1)
        time = [(trial_label,time) for time in forces.get("time")]
        pose_data = getTriangulatedPose(trial)
        # print("Before pose:",len(pose_data.get("x1")))
        pose_data = populatePoseGaps(pose_data, len(forces.get("time")))
        pose_data = addForcesToData(pose_data,forces)
        # print("After pose:",len(pose_data.get("x1")))
        # print("Force data:",len(forces.get("time")))
        trial_data.append(pd.DataFrame(pose_data, index=time).astype(float))

    return trial_data, pd.concat(trial_data)

def validateData(data):
    return np.isnan(np.sum(data))

data = loadData(train_path)
data_movements = formatTrialMovements(data)

movements = list(data_movements.keys())
movements.sort()
print(movements)
print()

train_trials = getMovementTrials("squat_jump", data_movements)
trial_data, df_for_scaler = collectFormatData(train_trials)

# LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
scaler = StandardScaler()
scaler = scaler.fit(df_for_scaler)
for idx in range(len(trial_data)):
    trial = trial_data[idx]
    trial_data[idx] = pd.DataFrame(data=scaler.transform(trial))


# print("Unnormalized:")
# print(df_for_scaler)
# print("Normalized trial 1 for training:")
# print(trial_data[0])

#Empty lists to be populated using formatted training data
trainX = []
trainY = []

n_future = 1
n_past = 12*5

#Reformat input data into a shape: (n_samples x timesteps x n_features)
print("FORMATTING DATA TO (n_samples x timesteps x n_features)")
for trial in trial_data:
    for idx in range(n_past, len(trial) - n_future +1):

        trainX.append(trial.loc[idx - n_past : idx-1, 0:51])
        trainY.append(trial.loc[idx, 52:57])


trainX, trainY, trial_data1 = np.array(trainX), np.array(trainY), np.array(df_for_scaler)
print('total trail data shape == {}.'.format(trial_data1.shape))
print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))
print()

if validateData(trainX) or validateData(trainY):
    print("ERROR, NaN DETECTED IN DATA")

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
history = model.fit(trainX, trainY, epochs=21, batch_size=64, validation_split=0.1, verbose=1)

model.save('model1')

plt.plot(history.history['loss'], label='Training loss')
plt.show()
plt.plot(history.history['val_loss'], label='Validation loss')
plt.show()
plt.legend()

#Perform inverse transformation to rescale back to original range
#Since we used 5 variables for transform, the inverse expects same dimensions
#Therefore, let us copy our values 5 times and discard them after inverse transform
# prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
# y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]

