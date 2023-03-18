import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    num_extra_points = int(forces_length/len(pose_data.get("x1")))

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
    


data = loadData(train_path)
data_movements = formatTrialMovements(data)

movements = list(data_movements.keys())
movements.sort()
print(movements)
print()

train_trials = getMovementTrials("squat_jump", data_movements)

trial_data = []
for trial in train_trials:
    forces = trial.get("grf")
    pose_data=getTriangulatedPose(trial)
    print("Before pose:",len(pose_data.get("x1")))
    #print("Pose data:",pose_data.get("x1"))
    pose_data = populatePoseGaps(pose_data, len(forces.get("time")))
    print("After pose:",len(pose_data.get("x1")))
    print("Force data:",len(forces.get("time")))
    trial_data.append(pd.DataFrame(pose_data))
    

