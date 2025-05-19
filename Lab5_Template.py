# Team attribution
# ArUco bin pose estimation: James Le, Prapti Bordoloi
# Tool pick and place: Prapti Bordoloi

"""
ME 5286 Spring 2024
Lab 5 Driver File Template

In this lab, students will use computer vision and machine learning 
to identify tools and their respective locations,
and use the UR5 robot to pick and place the identified tools in appropriate bins
The tools are laid out in a single line with known locations to the left side of the UR5
The bins are arranged in a 2x2 block placed at unknown coordinates in a random order on the right side of UR5

----------------------------------------------------------------------------------------------------

                                                [UR5 base]
-------------------------
|           |           |
|   bin 0   |   bin 3   |
|           |           |
-------------------------                          [Tool locations are known and fixed, order is not]
|           |           | [b]                          [tool 1]    [tool 2]    [tool 0]    [tool 3]
|   bin 2   |   bin 1   |
|           |           |
-------------------------   
[   bin locations are fixed wrt each other and reference [b]
    but order is uknown, and the coordinates of [b] are unknown ]
----------------------------------------------------------------------------------------------------

In this lab, there are two computer vision tasks combined with UR5 motions for
an image-guided pick-and-place task

Task 1: ArUco Marker Identifcation and Pose Estimation:
    - In this task, students will extract the pose of 
    the ArUco marker at the reference pose of the bin locations.

    - Further, students will use the ArUco markers inside each bin to identify the 
    order in which they are placed. 

    For example: All hammers should go to bin 0, but the location of bin 0 is not known,
    students will have to figure out the order in which the bins are arranged through
    aruco marker identification and using the pixel locations of markers wrt each other.
    Students will have the dimensions of the bin tray, and will estimate the approach pose
    for the bin tray through aruco pose estimation and for each bin . 

Task 2: Image classificaiton using Convolutional Neural Network:
    - In this task, students will train a Convolutional Neural Network using Tensorflow and Keras

    - In the lab, students will hover the wrist camera over each tool laid out in front of the UR5,
    identify the tool using the trained model,
    pick up the tool, and place it in the correct bin identified in Task 1.

Deliverables for Lab:
    - Video showing complete pick and place task with two sets of tools in random order.

"""

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import cv2
from cv2 import aruco as aruco
import time
import minimalmodbus
import numpy as np

from robolink import *
from robodk import *

# IMPORT YOUR OWN CLASSES/FUNCTIONS HERE
# e.g. from MyHelperFunctions import ArucoIdentifier, CustomNetwork, ImageProcessors etc

from utils.cnn_inference import cnn_inference
from utils.aruco_bin_mapper import estimate_marker_pose, map_markers_to_bins, compute_robot_bin_poses
from utils.instructor_provided.RobotiqGripper import RobotiqGripper

# Helper function to display an image
def show_image(frame):
    cv2.imshow('frame', frame)
    cv2.waitKey()
    cv2.destroyAllWindows()

# Helper function for gripper simulation
def simulate_gripper(action):
    print(f"Simulating gripper {action}")
    time.sleep(5)  # Sleep for 5 seconds as requested

# ------------ PARAMETERS YOU SHOULDN'T EDIT---------
# (Don't change these unless you absolutely know what you're doing)

# CAMERA PARAMETERS
CAMERA_FOCUS = 420.0
CAMERA_INDEX = 0
RESOLUTION = (640,480)

# ARUCO PARAMETERS
MARKER_LENGTH = 50.0 # [mm]
CAMERA_MATRIX_FILE = f'utils/instructor_provided/cameraMatrix_RobotiqWristCam_640x480.npy'
DISTORTION_COEFF_FILE = f'utils/instructor_provided/distCoeffs_RobotiqWristCam_640x480.npy'
cameraMatrix = np.load(CAMERA_MATRIX_FILE)
distCoeffs = np.load(DISTORTION_COEFF_FILE)
aruco_dict = aruco_dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50)  # Choose an ArUco dictionary ID
parameters = cv2.aruco.DetectorParameters()
font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text

#-----------------------------------------------------

# ------- PARAMETERS YOU CAN EDIT AS REQUIRED --------

# ROBOT PARAMETERS:
RUN_ON_ROBOT = True # change this to true when running this in lab

# GRIPPER PARAMETERS:
GRIPPER_PORT = 'COM5' # confirm this port before running [COM5 for Robot1, COM16 for Robot3]

# FULL PATH TO WHERE YOUR TRAINED MODEL IS STORED
MODEL_PATH = 'saved_models/tool_classifier_04_20_20_01.keras'

# edit this if your class labels are a different order
# or if you load them using your Cataloger functions
classes = {0: 'Hammer', 1: 'Pliers', 2: 'Screwdriver', 3: 'Wrench'}



#------------------------------------------------------

#--------------CODE STARTS HERE--------------------------#
# Feel free to edit the structure based on your code style
# Below is just a suggestion


# IMPORTANT!!!: RoboDK MUST have the correct station loaded!

## Initialize RoboDK
RDK = Robolink() 
robot = RDK.ItemUserPick('Select a robot', ITEM_TYPE_ROBOT)
if not robot.Valid():
    raise Exception('No robot selected or available')

if RUN_ON_ROBOT: 
    # if running on robot, connect to the robot

    success = robot.Connect()
    status, status_msg = robot.ConnectedState()
    if status != ROBOTCOM_READY:
        print(status_msg)
        raise Exception("Failed to connect: " + status_msg)
    RDK.setRunMode(RUNMODE_RUN_ROBOT)

# set robot parameters
joints_ref = robot.Joints()
target_ref = robot.Pose()
pos_ref = target_ref.Pos()
robot.setPoseFrame(robot.PoseFrame())
robot.setPoseTool(robot.PoseTool())
speed_linear = 100                          # Set linear speed in mm/s
speed_joint = 100                           # joint speed in deg/s
accel_linear = 100                         # linear accel in mm/ss
accel_joints= 100                            # joint accel in deg/ss
robot.setSpeed(speed_linear,speed_joint,accel_linear,accel_joints)    # Set linear and joint speed 

# gripper doesn't work on rdk simulation, so no need to call it when simulating
if RUN_ON_ROBOT: 
    # ## Initialize Gripper
    instrument = minimalmodbus.Instrument(GRIPPER_PORT, 9, debug = False)
    instrument.serial.baudrate = 115200
    gripper = RobotiqGripper(portname=GRIPPER_PORT,slaveaddress=9)
    # Activate gripper
    gripper.activate()
    print("Gripper activated and opened")
else:
    # Create a dummy gripper class for simulation
    # Code generated by Claude 3.7 Sonnet (Thinking) via Windsurf
    class DummyGripper:
        def closeGripper(self, *args, **kwargs):
            simulate_gripper("closing")
        def openGripper(self, *args, **kwargs):
            simulate_gripper("opening")
        def activate(self):
            print("Simulating gripper activation")
        def goTo(self, *args, **kwargs):
            print(f"Simulating gripper go to position {args[0] if args else 'unknown'}")
    
    gripper = DummyGripper()
    gripper.activate()

## Common Gripper Commands
# gripper.activate() #Turns on the gripper
# gripper.goTo(position=255,speed=255,force=255) # Commands the gripper to go to a specified position, with a specific speed and force
# gripper.closeGripper(speed=255,force=255) #Commands the gripper to close with specified speed and force
# gripper.openGripper(speed=255,force=255) #Commands the gripper to open with specified speed and force

# Initialize camera: again not needed if running simulation
    # You may define your own function for a neater look
if RUN_ON_ROBOT:
    cam = cv2.VideoCapture(CAMERA_INDEX)
    if not cam.isOpened():
        print("Cannot open camera {}".format(CAMERA_INDEX))
        exit()

    # Turn off Autofocus
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    # set camera parameters
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    cam.set(cv2.CAP_PROP_FOCUS, CAMERA_FOCUS)

# ## Common Camera & CV Functions
# # ret, frame = vidcap.read() # Captures what the camera currently sees. Ret sees if a frame is retrieved, frame is the current image
# # cv2.imshow(frame) # Shows the image stored in the tensor "frame"
# # cv2.imwrite('testimage.jpg',frame) # saves image captured in "frame" and saves it to "testimage.jpg"
# # image = cv2.imread('testimage.jpg') # reads image stored in file "testimage.jpg" and stores it in variable "image"
# # vidcap.release() # Releases the camera device
# # cv2.waitKey(1) # if waitKey delay is set to 0, it waits for user keypress indefinitely,
    # otherwise it waits for the specified delay time
# #cv.destroyAllWindows() # gets rid of all opened CV2 Windows
    

##~~~~~~~~~~~~~~~~~~~~Write Your Code After Here~~~~~~~~~~~~~~~~~~~~##

# Possible structure: 
    # Load stored model
    # Define Poses for:
        # tool locations
        # intermediate locations
        # bin relative translations
    # Move to general bin area for bin location sorting task:
        # find pose of marker ID 7
        # adjust robot pose based on aruco marker pose
        # move to the top of the bins
        # assign bin locations based on aruco marker locations
    # Begin pick and place task:
        # for each tool:
            # Move to tool area:
                # Move to tool camera location
                # identify tool
                # Pick up tool (move to pose above tool, move close to tool, close gripper, move to pose above tool)
            # Move to bin area:
                # Place tool in correct bin (move to pose above bin, move close to bin, open gripper, move to pose above bin)
    # Congratulate user on completing the task


# HOMOGENEOUS TRANSFORMATION MATRICES TO HELP YOU
    
#homeogeneous transformations you should use (feel free to change variable names)
original_tool_pose = transl(0,0,190)                        # TCP at gripper fingertips closed (from robot flange)
flange_to_camera = transl(0,43.75,10)*rotx(np.deg2rad(-30)) # camera frame w.r.t flange frame
tcp_to_camera = invH(original_tool_pose)*flange_to_camera   # camera frame w.r.t gripper frame
camera_to_tcp = invH(flange_to_camera)*original_tool_pose   # gripper fingertips w.r.t camera frame

# All of the poses provided below have the gripper facing down (opposite the UR5 z-axis)
# you will need to transform most of these pose to have the camera facing the tool/bin
# when identifying tools or markers
# [HINT: use one of the transformation matrices from above]

# -- tool locations
tool_0 = RDK.Item('tool 0').Pose() # if you use the provided RDK Station
tool_relative_x_coordinates = [0, -225, -450, -640] # w.r.t tool 0

# --bin locations

# pose at which to look for aruco marker id 7
# you will need to transform this pose so that the camera is looking at the marker
bin_reference = RDK.Item('bin reference').Pose()   

# pose at which to start identifying the markers inside the bins
above_bins_offset = [200, 0, -100] # translation offset w.r.t corrected bin reference

# location of bins wrt the "corrected_bin_reference" pose.
# the array below is ordered with the bins counting anticlockwise from 0 at the lower left location
# 
# ---------------
# |  3  |   2   |
# ---------------
# |  0  |   1   |
# ---------------
bin_relative_translations = [   [250, -75, 225],
                                [125, -75, 225],
                                [125, 200, 225],
                                [250, 200, 225]
                            ] # because robodk plays nicer with lists
# you will need to sort these or create a different sorted list based on 
# the order of the aruco markers that are inside these bins


#~~~~Example implementation~~~~

# Homogeneous transforms for TCP and camera
original_tool_pose = transl(0,0,190)
flange_to_camera = transl(0,43.75,10)*rotx(np.deg2rad(-30))
tcp_to_camera = invH(original_tool_pose)*flange_to_camera
camera_to_tcp = invH(flange_to_camera)*original_tool_pose

# Define tool poses
tool_0 = RDK.Item('tool 0').Pose()
tool_poses = [tool_0*transl(x,0,0) for x in tool_relative_x_coordinates]

# Map specific tool images for simulation
tool_image_paths = {
    0: 'pictures/hammer.jpg',
    1: 'pictures/pliers.jpg',
    2: 'pictures/screwdriver.jpg',
    3: 'pictures/wrench.jpg'
}

# Map tool labels to bin indices
tool_label_to_index = {v:k for k,v in classes.items()}

# --- Bin tray calibration ---
bin_reference_joints = [23.122886, -82.230903, 57.897172, -65.666268, -90.000000, -66.877114]
home = [0.000000, -90.000000, 0.000000, -90.000000, 0.000000, 0.000000]
robot.MoveJ(home)
robot.MoveJ(bin_reference_joints)
robot.MoveJ(bin_reference*camera_to_tcp)
time.sleep(0.5)
if RUN_ON_ROBOT:
    ret, frame = cam.read()
    ret, frame = cam.read()
    time.sleep(7)
    ret, frame = cam.read()
    cv2.imwrite("ArUco Reference.jpg", frame)
    print(f"Saved ArUco reference marker.")
    if not ret: raise Exception("Failed to capture frame for bin calibration")
else:
    frame = cv2.imread('pictures/aruco_picture.jpg')
    print("Loaded reference ArUco marker image")

corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
if ids is None or 7 not in ids.flatten(): raise Exception("Reference ArUco marker ID 7 not found")
idx = list(ids.flatten()).index(7)
rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers([corners[idx]], MARKER_LENGTH, cameraMatrix, distCoeffs)
rotM, _ = cv2.Rodrigues(rvecs[0][0])
pose_marker = np.eye(4); 
pose_marker[0,3] = tvecs[0][0][0]
pose_marker[1,3] = tvecs[0][0][1]
pose_marker_mat = Mat(pose_marker.tolist())
corrected_bin_reference = bin_reference * invH(pose_marker_mat)
robot.MoveJ(corrected_bin_reference*camera_to_tcp)
robot.MoveJ(corrected_bin_reference*transl(above_bins_offset)*camera_to_tcp)
time.sleep(0.5)

# Detect bin markers
if RUN_ON_ROBOT:
    time.sleep(3)
    ret, frame = cam.read()
    ret, frame = cam.read()
    ret, frame = cam.read()
    cv2.imwrite("top_of_bins.jpg", frame)
    print(f"Saved top of bins pic.")
    if not ret: raise Exception("Failed to capture frame for bin detection")
else:
    frame = cv2.imread('pictures/top_of_bins.jpg')
    print("Loaded bin ArUco markers image")
corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
if ids is None:
    raise Exception("No bin markers detected")

marker_positions = []

for i, mid in enumerate(ids.flatten()):
    c = corners[i][0]; cx, cy = c[:,0].mean(), c[:,1].mean()
    marker_positions.append((int(mid), cx, cy))
# Sort by image row (y) then column (x)
marker_positions.sort(key=lambda m:(m[2],m[1]))

bin_flange_poses = {}
for i, (mid,_,_) in enumerate(marker_positions):
    rel = bin_relative_translations[i]
    bin_flange_poses[mid] = corrected_bin_reference*transl(*rel)

# --- Pick and place loop ---
print("\n--- Starting pick and place operations ---\n")
for i, tool_pose in enumerate(tool_poses):
    print(f"\nProcessing tool {i}...")
    # Position camera above tool
    cam_flange = tool_pose * camera_to_tcp
    robot.MoveL(cam_flange)
    time.sleep(0.5)
    if RUN_ON_ROBOT:
        # Discard first frame which is usually too bright
        ret, _ = cam.read()
        time.sleep(1)
        ret, frame = cam.read()
        if not ret: raise Exception("Failed to capture frame for tool detection")
    else:
        frame = cv2.imread(tool_image_paths[i])
        print(f"Loaded image for tool {i}: {tool_image_paths[i]}")
        show_image(frame)

    # Identify tool
    label = cnn_inference(frame)
    print(f"Detected tool: {label}")
    tid = tool_label_to_index[label]

    # Save detected image
    ts = time.strftime("%m%d_%H%M")
    output_path = f"{label}_{ts}.jpg"
    cv2.imwrite(output_path, frame)
    print(f"Saved detection image to {output_path}")

    # Pick up tool
    pick_flange = tool_pose * original_tool_pose * transl(0, 0, 180)
    robot.MoveL(pick_flange)
    if RUN_ON_ROBOT:
        gripper.closeGripper()
        time.sleep(0.5)
    else:
        simulate_gripper("closing")

    # Lift tool
    lift_pose = pick_flange * transl(0,0,-100)
    robot.MoveL(lift_pose)
    print(f"Picked up {label}, moving to bin {tid}")

    # Move to bin and place tool
    place_flange = bin_flange_poses[tid]
    robot.MoveL(place_flange)
    drop_pose = place_flange * transl(0,0,50)
    robot.MoveL(drop_pose)
    if RUN_ON_ROBOT:
        gripper.openGripper()
        time.sleep(0.5)
    else:
        simulate_gripper("opening")
    print(f"Placed {label} in bin {tid}")

    # Retract
    robot.MoveL(place_flange)

print("Returning to home position")
robot.MoveJ(home)

print("All tools sorted. Congratulations!")

# Cleanup
if RUN_ON_ROBOT:
    cam.release()
    gripper.openGripper()