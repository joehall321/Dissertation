import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras import optimizers
from DataFormatter import DataFromatter
import scipy.signal as ss
from heapq import nlargest
import math


def getSampleSize(model_name):
    idx = model_name.find("ss")+2
    size = ""
    while model_name[idx]!='_':
        size+=model_name[idx]
        idx+=1
    return int(size)

def plotTrialEndLines(axis, trial_lengths):
    for end in trial_lengths:
        axis.axvline(x = end, color = 'r')

def plotPredictions(model_name, movement, truth, prediction, trial_lengths):

    # Attaching 2D axis to the figure
    fig, axis = plt.subplots(2, 3)
    fig.suptitle(model_name+"\n"+str(len(trial_lengths))+" "+movement+" trials")
    axis[0,0].title.set_text('ground_force1_vx')
    axis[0,0].plot(truth[51], label='Truth')
    axis[0,0].plot(prediction[51], label='LSTM prediction')
    t_peaks, t_mags, p_peaks, p_mags = extractPeaks(3,truth[51], prediction[51], trial_lengths)
    axis[0,0].scatter(t_peaks, t_mags, label="Truth Peaks")
    axis[0,0].scatter(p_peaks, p_mags, label="Prediction Peaks")
    axis[0,0].legend()
    plotTrialEndLines(axis[0,0],trial_lengths)

    axis[0,1].title.set_text('ground_force1_vy')
    axis[0,1].plot(truth[52], label='Truth')
    axis[0,1].plot(prediction[52], label='LSTM prediction')
    t_peaks, t_mags, p_peaks, p_mags = extractPeaks(3,truth[52], prediction[52], trial_lengths)
    axis[0,1].scatter(t_peaks, t_mags, label="Truth Peaks")
    axis[0,1].scatter(p_peaks, p_mags, label="Prediction Peaks")
    axis[0,1].legend()
    plotTrialEndLines(axis[0,1],trial_lengths)

    axis[0,2].title.set_text('ground_force1_vz')
    axis[0,2].plot(truth[53], label='Truth')
    axis[0,2].plot(prediction[53], label='LSTM prediction')
    t_peaks, t_mags, p_peaks, p_mags = extractPeaks(3,truth[53], prediction[53], trial_lengths)
    axis[0,2].scatter(t_peaks, t_mags, label="Truth Peaks")
    axis[0,2].scatter(p_peaks, p_mags, label="Prediction Peaks")
    axis[0,2].legend()
    plotTrialEndLines(axis[0,2],trial_lengths)

    axis[1,0].title.set_text('ground_force2_vx')
    axis[1,0].plot(truth[54], label='Truth')
    axis[1,0].plot(prediction[54], label='LSTM prediction')
    t_peaks, t_mags, p_peaks, p_mags = extractPeaks(3,truth[54], prediction[54], trial_lengths)
    axis[1,0].scatter(t_peaks, t_mags, label="Truth Peaks")
    axis[1,0].scatter(p_peaks, p_mags, label="Prediction Peaks")
    axis[1,0].legend()
    plotTrialEndLines(axis[1,0],trial_lengths)

    axis[1,1].title.set_text('ground_force2_vy')
    axis[1,1].plot(truth[55], label='Truth')
    axis[1,1].plot(prediction[55], label='LSTM prediction')
    t_peaks, t_mags, p_peaks, p_mags = extractPeaks(3,truth[55], prediction[55], trial_lengths)
    axis[1,1].scatter(t_peaks, t_mags, label="Truth Peaks")
    axis[1,1].scatter(p_peaks, p_mags, label="Prediction Peaks")
    axis[1,1].legend()
    plotTrialEndLines(axis[1,1],trial_lengths)

    axis[1,2].title.set_text('ground_force2_vz')
    axis[1,2].plot(truth[56], label='Truth')
    axis[1,2].plot(prediction[56], label='LSTM prediction')
    t_peaks, t_mags, p_peaks, p_mags = extractPeaks(3,truth[56], prediction[56], trial_lengths)
    axis[1,2].scatter(t_peaks, t_mags, label="Truth Peaks")
    axis[1,2].scatter(p_peaks, p_mags, label="Prediction Peaks")
    axis[1,2].legend()
    plotTrialEndLines(axis[1,2],trial_lengths)

    # Set common labels
    fig.text(0.5, 0.04, 'Frames', ha='center', va='center')
    fig.text(0.06, 0.5, 'Force (N)', ha='center', va='center', rotation='vertical')
    plt.show()

def extractPeaks(k, truths, preds, trial_lengths):

    # Get truth peaks
    t_idxs, t_mags = [],[]
    for idx in range(len(trial_lengths)):
        if idx==0:
            trial_forces = truths.iloc[:trial_lengths[idx]]
        else:
            trial_forces = truths.iloc[trial_lengths[idx-1]:trial_lengths[idx]]
        squared_forces = trial_forces**2
        peaks = ss.find_peaks(squared_forces,distance=25)[0] 
        fs = nlargest(k,[trial_forces.iloc[idx] for idx in peaks])
        t_idxs.extend([truths[truths == f].index[0] for f in fs])
        t_mags.extend(fs)

    # Find local prediction peaks to all truth peaks
    p_idxs, p_mags = [], []
    window = 10
    counter_idx=0
    for idx in t_idxs:
        start = idx-window
        end = idx+window
        if start<0: start=0
        fs = preds[start:end]
        if t_mags[counter_idx]<0:
            peak = min(fs)
        else:
            peak = max(fs)
        p_idx = fs[fs == peak].index[0]
        p_max = fs[p_idx]

        p_idxs.append(p_idx)
        p_mags.append(p_max)
        counter_idx+=1
    
    return t_idxs, t_mags, p_idxs, p_mags


def kpeaks(k, truths, preds, trial_lengths):
    k_peaks = []
    for idx in range(51,57):
        t_peaks, t_mags, p_peaks, p_mags = extractPeaks(k,truths[idx], preds[idx], trial_lengths)
        xys = (np.subtract(t_peaks,p_peaks)**2) + (np.subtract(t_mags,p_mags)**2)
        k_p = sum([math.sqrt(xy) for xy in xys])/len(xys)

        k_peaks.append(k_p)

    results = {"Fx":sum([k_peaks[0],k_peaks[3]])/2,"Fy":sum([k_peaks[1],k_peaks[4]])/2,
               "Fz":sum([k_peaks[2],k_peaks[5]])/2}
    avg = sum(k_peaks)/len(k_peaks)
    return results, avg

def rmse(truths, preds):
    errors = []
    for idx in range(51,57):
        errors.append(mean_squared_error(truths[idx], preds[idx], squared=False))

    results = {"Fx":sum([errors[0],errors[3]])/2,"Fy":sum([errors[1],errors[4]])/2,
               "Fz":sum([errors[2],errors[5]])/2}
    avg = sum(errors)/len(errors)
    return results, avg

test_path = "Datasets/force_pose/val.json"
movement = 'all'
model_names = ["all-ss20_blstm_64_1024_dense256_lr1e-4_bs64_eps50"]
plot_predictions=True

RMSEs, peaks1, peaks2, peaks3 = [],[],[],[]

for model_name in model_names:

    sample_size = getSampleSize(model_name)

    data_formatter = DataFromatter()

    data = data_formatter.loadData(test_path)
    data_movements = data_formatter.formatTrialMovements(data, flip_xz_flag=True)

    print()
    print("Movements:")
    print(data_formatter.dict_dims(data_movements))
    print()

    model = load_model("Models/"+model_name)
    print()
    print("Model: ",model_name)

    for movement in {"SingleLegSquatR"}:
        # Get list of trials that are doing a specified movement
        trials = data_formatter.getMovementTrials(movement, data_movements)[0:3]
        print()
        print("Validating "+str(len(trials))+" "+movement+" trials")

        # Trial data has shape (number of force data points, (17*3d keypoints + 2*3d forces + subject mass)
        # 3D keypoints are expanded by incremental difference as force captured @600hz compared to keypoints @50hz
        # This needs to be changed as we are increasing noise in HPE. 
        trial_data = data_formatter.collectFormatData(trials,"linearinterpolation" in model_name)

        # print("Before normalization trial data:")
        # print(trial_data[0].iloc[59:69])
        # Values need to be normalized using the same scaler as training
        # normalize the dataset
        trial_data = data_formatter.NormaliseData(trial_data, model_name)

        # print("Unnormalized:")
        # print(df_for_scaler)
        # print("Normalized trial 1 for training:")
        # print(trial_data[0])

        #Reformat input data into a shape: (n_samples x timesteps x n_features)
        # Forming each x and y by sliding window of n_past data points   
        # Each trail is seperated in training data by sliding window
        print("Sample size:",sample_size)
        testX, testY, trial_lengths = data_formatter.formatDataSamples(trial_data, sample_size, "endframeforce" in model_name)

        # Make prediction
        pred_y = model.predict(testX)

        # print()
        # print("Shape of val data:",testX.shape)
        # print("Shape of prediction data:",pred_y.shape)

        # print("Normalized Prediction:",normalized_prediction)
        # print("Normalized Truth:",testY)

        #Perform inverse transformation to rescale back to original range
        #Since we used 5 variables for transform, the inverse expects same dimensions
        #Therefore, let us copy our values 5 times and discard them after inverse transform
        prediction = pd.DataFrame(index=range(pred_y.shape[0]),columns=range(trial_data[0].shape[1]))
        truth = pd.DataFrame(index=range(pred_y.shape[0]),columns=range(trial_data[0].shape[1]))
        # Replace force columns in copy with normalized model prediction columns
        prediction.loc[:, 51:56] = pred_y
        truth.loc[:, 51:56] = testY

        avg_mass = data_formatter.avg_mass
        prediction = pd.DataFrame(data_formatter.inverseNormalize(prediction, model_name)).loc[:,51:56]*avg_mass
        truth = pd.DataFrame(data_formatter.inverseNormalize(truth, model_name)).loc[:,51:56]*avg_mass

        if plot_predictions:
            plotPredictions(model_name, movement, truth, prediction, trial_lengths)

        print("Movement:",movement)
        # RMSE error
        rmse_errors, rmse_avg= rmse(truth,prediction)
        print("RMSE accuracy:",rmse_avg)
        RMSEs.append(rmse_avg)

        for label in rmse_errors:
            print(label,": ",rmse_errors[label])
        print()
        
        # K-peaks error
        peaks=[]
        for k in range(1,4):
            k_peaks, k_peaks_avg = kpeaks(k,truth,prediction, trial_lengths)
            peaks.append(k_peaks_avg)
            if k==1:
                peaks1.append(k_peaks_avg)
            if k==2:
                peaks2.append(k_peaks_avg)
            if k==3:
                peaks3.append(k_peaks_avg)
            print(str(k)+"-peaks accuracy:",k_peaks_avg)
            for label in k_peaks:
                print(label,": ",k_peaks[label])
            print()
        print("1,2,3 k-peaks avg accuracy:",sum(peaks)/len(peaks))