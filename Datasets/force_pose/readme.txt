Due to privacy and rights concerns, the RGB data is not provided as part of the data release. However, the detected 2D points, motion capture points, and force plate data are publicly available.

The ForcePose dataset contains 168 movements and 8 subjects (6 train, 2 validation).

124 trials/movements in train
44 trials/movements in validation

Training data force plate ranges (fx1, fy1, fz1, fx2, fy2, fz2)
(Newtons)
Min train forces: [-271.711, -0.0, -405.757, -272.599, -0.0, -378.167]
Max train forces: [256.854, 2870.934, 591.252, 201.034, 3573.284, 570.624]

(Newtons/mass)
Min train forces: [-3.264, -0.0, -4.635, -3.114, -0.0, -4.320]
Max train forces: [3.085, 34.486, 6.866, 2.335, 32.362, 6.854]

Each sample contains multiple camera views (up to 8), 2D pose predictions on each view,
triangulated poses for each time step, and corresponding Ground Reaction Forces (GRFs)

Subject Masses:
Subject1 : 83.25
Subject2 : 86.48
Subject3 : 87.54
Subject4 : 86.11
Subject5 : 74.91
Subject6 : 111.91
Subject7 : 82.64
Subject8 : 90.44

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

Data format (list):
Each video/trial is a dict with the following keys:

- "subject" (string)
- "movement" (string)
- "frame_width" (int)
- "frame_height" (int)
- "total_frames" (int)
- "frames" (list)
	- "image_name" (string)
	- "image_index" (int)
	- "cam"_[x] (dict)
		- 'box' (bounding box - xtl,ytl,xbr,ybr)
		- 'keypoints' - 51 2D joints (17 joints x 3 [x,y,confidence]) (MSCOCO format)
	- "triangulated_pose" (list)
		- 17 triangulated joints (MSCOCO format)
- "mocap" (dict)
	- [marker_name] (list)
		- N frames, marker position  
- "grf" (dict)
	- 'time' (list)
	- 'ground_force1_vx' (list)
	- 'ground_force1_vy' (list)
	- 'ground_force1_vz' (list)
	- 'ground_force2_vx' (list)
	- 'ground_force2_vy' (list)
	- 'ground_force2_vz' (list)

Please cite our work if you need this dataset and publication useful.
