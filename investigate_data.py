import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

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

skeleton = [(17,15),(15,13),(13,7),(7,9),(7,6),(9,11),
(6,12),(6,8),(8,10),(12,13),(12,14),(14,16)]

# Helper functions to fetch data and format data

def loadData():
    train_data = "Datasets/force_pose/train.json"
    data = json.load(open(train_data))
    return data

def getVideoTrial(index):
    return loadData()[index]

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
def update_3D_triangulated_points(num_steps, frames, xyz_points, ax):
    traingulated = frames[num_steps].get("triangulated_pose")
    xs = [point[0] for point in traingulated]
    ys = [point[1] for point in traingulated]
    zs = [point[2] for point in traingulated]

    xyz_points._offsets3d = (xs, ys, zs)

def plot3DTriangulatedPoints(frames):
    num_steps = 311

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

    # Creating the Animation object
    ani = animation.FuncAnimation(
        fig, update_3D_triangulated_points, num_steps, fargs=(frames, xyz_points, ax),interval=100)
    plt.show()

# Functions to show animations
def showMocap(index):
    video_trial = getVideoTrial(index)
    mocap_data = getMOCAP(video_trial)
    plotMocap(mocap_data)

def showTriangulated(index):
    video_trial = getVideoTrial(index)
    frames = getFrames(video_trial)
    plot3DTriangulatedPoints(frames)

showTriangulated(0)