import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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
    space=1.25
    return ((min(x),max(x)),(min(y),max(y)),(min(z),max(z)))

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
        line, = ax.plot([], [], [], lw=2)
        lines.append(line)

    # Creating the Animation object
    ani = animation.FuncAnimation(
        fig, update_3D_triangulated_points, num_steps, fargs=(frames, xyz_points, lines),interval=100)
    plt.show()

# Functions to show animations
def showMocap(video_trial):
    mocap_data = getMOCAP(video_trial)
    plotMocap(mocap_data)

def showTriangulated(video_trial):
    frames = getFrames(video_trial)
    plot3DTriangulatedPoints(frames)

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
showTriangulated(video_trial)