import json
import matplotlib.pyplot as plt

train_data = "Datasets/force_pose/train.json"
mocap_markers = ['CLAV', 'LACRM', 'LASIS', 'LHEEL', 'LLAK',
               'LLEB', 'LLFARM', 'LLHND', 'LLKN', 'LLTOE',
               'LLWR', 'LMAK', 'LMEB', 'LMFARM', 'LMHND',
               'LMKN', 'LMTOE', 'LMWR', 'LPSI', 'LSHANK',
               'LTHIGH', 'LUPARM', 'RACRM', 'RASIS',
               'RHEEL', 'RLAK', 'RLEB', 'RLFARM', 'RLHND',
               'RLKN', 'RLTOE', 'RLWR', 'RMAK', 'RMEB', 'RMFARM',
               'RMHND', 'RMKN', 'RMTOE', 'RMWR', 'RPSI',
               'RSHANK', 'RTHIGH', 'RUPARM', 'STRM',
               'T1', 'T10', 'THEAD']

data = json.load(open(train_data))
print(data[0].keys())
mocap_data = data[0].get("mocap")
frames = data[0].get("frames")
grf_data = data[0].get("grf")
print(data[0].get("total_frames"))
print(len(frames))
print(len(grf_data["ground_force1_vx"]))
print(len(mocap_data['CLAV']))