import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.patches as mpatches

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

cameras = ['cam_17400883', 'cam_17400877', 'cam_17400884', 'cam_17400879', 
'cam_17400881', 'cam_17400878', 'cam_17364068', 'cam_17400880']

skeleton = [[16,14,12,6,8,10],[15,13,11,5,7,9],[12,11],[5,6,4,3,5]]

# Helper functions to fetch data and format data
def loadData():
    data = json.load(open(train_data))
    return data

def getMovement(data):
    return data.get("Movement")

def getFrames(data):
    return data.get("frames")

def getMOCAP(data):
    return data.get("mocap")

def getMOCAP(data):
    return data.get("mocap")

def getGRFs(data):
    return data.get("grf")

def formatMocapMarkerXYZ(mocap_data):
    data = {}
    for marker in mocap_data:
        data[marker]=[]
        data[marker].append([point[0] for point in mocap_data[marker]])
        data[marker].append([point[1] for point in mocap_data[marker]])
        data[marker].append([point[2] for point in mocap_data[marker]])
    return data

# Functions to find xyz limits for 3D graphs
def getMocapLimits(mocap_marker_data):
    x = []
    y = []
    z = []
    for marker in mocap_marker_data:
        x.append(max(mocap_marker_data[marker][0]))
        x.append(min(mocap_marker_data[marker][0]))
        y.append(max(mocap_marker_data[marker][1]))
        y.append(min(mocap_marker_data[marker][1]))
        z.append(max(mocap_marker_data[marker][2]))
        z.append(min(mocap_marker_data[marker][2]))
    return ((min(x),max(x)),(min(y),max(y)),(min(z),max(z)))

def getTriangulatedLimits(frames):
    x = []
    y = []
    z = []
    for frame in frames:
        traingulated = frame.get("triangulated_pose")
        xs = [points[0] for points in traingulated]
        ys = [points[1] for points in traingulated]
        zs = [points[2] for points in traingulated]
        x.append(max(xs))
        x.append(min(xs))
        y.append(max(ys))
        y.append(min(ys))
        z.append(max(zs))
        z.append(min(zs))
    return ((min(x),max(x)),(min(y),max(y)),(min(z),max(z)))

def get2DKeypointLimits(frames, cam):
    x = []
    y = []
    for frame in frames:
        keypoints = frame.get(cam).get("keypoints")
        xs=[]
        ys=[]
        for idx in range(0,len(keypoints),3):
            xs.append(keypoints[idx])
            ys.append(keypoints[idx+1])
        x.append(max(xs))
        x.append(min(xs))
        y.append(max(ys))
        y.append(min(ys))
    return ((min(x),max(x)),(min(y),max(y)))

############################################################################

# Update function used for animation of 3D scatter
def update_mocap_points(num_steps, mocap_marker_data, xyz_points, ax):
    x=[]
    y=[]
    z=[]
    for marker in mocap_marker_data:
        markers.append(marker)
        x.append(mocap_marker_data[marker][0][num_steps])
        y.append(mocap_marker_data[marker][1][num_steps])
        z.append(mocap_marker_data[marker][2][num_steps])
    xyz_points._offsets3d = (x, y, z)

def plotMocap(mocap_data):
    num_steps = 621

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Create x,y,z points initially without data
    xyz_points = ax.scatter([], [], [])

    mocap_marker_data = formatMocapMarkerXYZ(mocap_data)
    xyz_limits = getMocapLimits(mocap_marker_data)

    # Setting the axes properties
    ax.set(xlim3d=(xyz_limits[0][0], xyz_limits[0][1]), xlabel='X')
    ax.set(ylim3d=(xyz_limits[1][0], xyz_limits[1][1]), ylabel='Y')
    ax.set(zlim3d=(xyz_limits[2][0], xyz_limits[2][1]), zlabel='Z')

    # Creating the Animation object
    ani = animation.FuncAnimation(
        fig, update_mocap_points, num_steps, fargs=(mocap_marker_data, xyz_points, ax),interval=100)
    plt.show()

# Update function used for animation of 3D scatter
def update_3D_triangulated_points(num_steps, frames, xyz_points, lines):
    traingulated = frames[num_steps].get("triangulated_pose")
    xs = [point[0] for point in traingulated]
    ys = [point[1] for point in traingulated]
    zs = [point[2] for point in traingulated]

    xyz_points._offsets3d = (xs, ys, zs)
    for idx in range(len(skeleton)):
        lines[idx].set_data([xs[idx]for idx in skeleton[idx]], [ys[idx]for idx in skeleton[idx]])
        lines[idx].set_3d_properties([zs[idx]for idx in skeleton[idx]])

def plot3DTriangulatedPoints(frames):
    num_steps = len(frames)

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Create x,y,z points initially without data
    xyz_points = ax.scatter([], [], [])
    xyz_limits = getTriangulatedLimits(frames)

    # Setting the axes properties
    ax.set(xlim3d=(xyz_limits[0][0], xyz_limits[0][1]), xlabel='X')
    ax.set(ylim3d=(xyz_limits[1][0], xyz_limits[1][1]), ylabel='Y')
    ax.set(zlim3d=(xyz_limits[2][0], xyz_limits[2][1]), zlabel='Z')

    # Plot floor mesh
    X, Z = np.meshgrid(np.arange(xyz_limits[0][0], xyz_limits[0][1]), 
                       np.arange(xyz_limits[2][0], xyz_limits[2][1]))
    Y = 0*X
    ax.plot_surface(X, Y, Z, alpha=0.5)  # the horizontal plane

    # Create skeleton lines
    lines=[]
    for _ in skeleton:
        line, = ax.plot([], [], [], lw=2, color="cornflowerblue")
        lines.append(line)

    # Creating the Animation object
    ani = animation.FuncAnimation(
        fig, update_3D_triangulated_points, num_steps, fargs=(frames, xyz_points, lines),interval=100)
    plt.show()

# Update function used for animation of 2D GRF graph
def update_grf_timeline(num_steps, xs, ys, line):
    x = xs[num_steps]
    line.set_data([x for y in ys],ys)

def plotGRFs(grfs):
    num_steps = len(grfs.get("time"))

    # Attaching 2D axis to the figure
    fig, axis = plt.subplots(3, 1)

    grf_labels=['ground_force1_vx','ground_force1_vy','ground_force1_vz']
    xs = grfs.get("time")
    
    animations=[]

    counter = 0
    for label1 in grf_labels:
        ax = axis[counter]
        counter +=1

        label2 = label1.replace("1", "2")
        ys1 = grfs.get(label1)
        ys2 = grfs.get(label2)

        # Create x,y points initially without data
        points1 = ax.scatter([xs], [ys1], s=1, color="blue")
        points2 = ax.scatter([xs], [ys2], s=1, color="cornflowerblue")

        # Timeline
        line, = ax.plot([], [], lw=2, color="red")
        y_lims=ax.get_ylim()
        ys = [int(y_lims[0]),int(y_lims[1])]

        y1 = mpatches.Patch(color='blue', label=label1)
        y2 = mpatches.Patch(color='cornflowerblue', label=label2)
        ax.legend(handles=[y1,y2])

        ax.set_title(label1.replace("1", ""))

        # Creating the Animation object
        ani = animation.FuncAnimation(
            fig, update_grf_timeline, num_steps, fargs=(xs, ys, line))
        animations.append(ani)

    plt.xlabel("Seconds")
    plt.ylabel("GRFs / Newtons")
    plt.show()

# Update function used for animation of 2D scatter
def update_2D_keypoints(num_steps, frames, xy_points, camera):
    keypoints = frames[num_steps].get(camera).get("keypoints")
    xs=[]
    ys=[]
    for idx in range(0,len(keypoints),3):
        xs.append(keypoints[idx])
        ys.append(keypoints[idx+1])
    xyz_points._offsets3d = (xs, ys)

def plot2DKeypoints(frames):
    num_steps = len(frames)

    camera='cam_17400883'

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = fig.add_subplot()

    # Create x,y,z points initially without data
    xy_points = ax.scatter([], [])
    xy_limits = get2DKeypointLimits(frames,camera)

    # Setting the axes properties
    print(xy_limits)
    plt.xlim([xy_limits[0][0],xy_limits[0][1]])
    plt.ylim([xy_limits[1][0],xy_limits[1][1]])

    # Create skeleton lines
    # lines=[]
    # for _ in skeleton:
    #     line, = ax.plot([], [], lw=2, color="cornflowerblue")
    #     lines.append(line)

    # Creating the Animation object
    ani = animation.FuncAnimation(
        fig, update_2D_keypoints, num_steps, fargs=(frames, xy_points, camera),interval=100)
    plt.show()

# Functions to show animations
def showMocap(video_trial):
    mocap_data = getMOCAP(video_trial)
    plotMocap(mocap_data)

def showTriangulated(video_trial):
    frames = getFrames(video_trial)
    plot3DTriangulatedPoints(frames)

def showGRFs(video_trial):
    grfs = getGRFs(video_trial)
    plotGRFs(grfs)

def show2DKeypoints(video_trial):
    frames = getFrames(video_trial)
    plot2DKeypoints(frames)

# Fetch first trial data seen for specified movement
def getFirstTrialMovement(movement):
    data = loadData()
    movements={}
    for video_trial in data:
        move = video_trial.get("movement")
        movements[move]=None
        if move==movement:
            return video_trial
    print("Could not find movement. Here are all movements: ",
          list(movements.keys()))
    exit()

video_trial = getFirstTrialMovement("Squat_Jump_03")
show2DKeypoints(video_trial)