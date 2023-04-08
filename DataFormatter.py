import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataFromatter():

    def __init__(self) -> None:
        self.subject_masses= {
        "Subject1" : 83.25,
        "Subject2" : 86.48,
        "Subject3" : 87.54,
        "Subject4" : 86.11,
        "Subject5" : 74.91,
        "Subject6" : 111.91,
        "Subject7" : 82.64,
        "Subject8" : 90.44 }

    # Load data
    def loadData(self, path):
        data = json.load(open(path))
        return data

    # Categrises each trial into movement list 
    def formatTrialMovements(self, train_data):
        train_movements = {}
        for trial in train_data:
            movement=trial.get("movement")
            train_movements.setdefault(movement,[])
            train_movements.get(movement).append(trial)
        return train_movements

    # Returns joined list of trials for specified movement
    def getMovementTrials(self, movement, data):
        movement_data = []
        for type in data:
            if movement in type.lower():
                movement_data.extend(data[type])
        return movement_data

    # Return dictonary of {x1:[x1,...,xt], y1:[y1,...,yt], z1:[z1,...,zt],
    #                   ..., x17:[x1,...,xt],y17:[y1,...,yt], z17:[z1,...,zt]}
    def getTriangulatedPose(self, trial):
        frames = trial.get("frames")
        mass=self.subject_masses.get(trial.get("subject"))

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
    def populatePoseGaps(self, pose_data, forces_length):
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
        
    def addForcesToData(self, data, forces):
        for force_label in forces:
            if force_label!="time":
                data[force_label] = forces[force_label]
        return data

    # Collects and formats data
    # creates a list of dataframes 
    def collectFormatData(self, train_trials):
        trial_data = []

        for idx in range(len(train_trials)):
            trial = train_trials[idx]
            forces = trial.get("grf")
            trial_label = "trail_"+str(idx+1)

            # Get time to be used as data frame index
            time = [(trial_label,time) for time in forces.get("time")]

            pose_data = self.getTriangulatedPose(trial)
            # print("Before pose:",len(pose_data.get("x1")))
            pose_data = self.populatePoseGaps(pose_data, len(forces.get("time")))
            pose_data = self.addForcesToData(pose_data,forces)
            # print("After pose:",len(pose_data.get("x1")))
            # print("Force data:",len(forces.get("time")))
            trial_data.append(pd.DataFrame(pose_data, index=time).astype(float))

        return trial_data, pd.concat(trial_data)

    def validateData(self,data):
        return np.isnan(np.sum(data))
    
    def formatDataSamples(self, trial_data):
        #Empty lists to be populated using formatted training data
        trainX = []
        trainY = []

        n_future = 1
        n_past = 12*5

        #Reformat input data into a shape: (n_samples x timesteps x n_features)
        # Forming each x and y by sliding window of n_past data points   
        # Each trail is seperated in training data by sliding window
        print("FORMATTING DATA TO (n_samples x timesteps x n_features)")
        for trial in trial_data:
            for idx in range(n_past, len(trial) - n_future +1):

                trainX.append(trial.loc[idx - n_past : idx-1, 0:51])
                trainY.append(trial.loc[idx, 52:57])


        trainX, trainY = np.array(trainX), np.array(trainY)

        if self.validateData(trainX) or self.validateData(trainY):
            print("ERROR, NaN DETECTED IN DATA")

        return trainX, trainY