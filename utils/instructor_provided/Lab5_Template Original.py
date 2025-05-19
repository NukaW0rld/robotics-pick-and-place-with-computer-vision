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

from RobotiqGripper import*
from robolink import *
from robodk import *

# IMPORT YOUR OWN CLASSES/FUNCTIONS HERE
# e.g. from MyHelperFunctions import ArucoIdentifier, CustomNetwork, ImageProcessors etc

from utils import *


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
RUN_ON_ROBOT = False # change this to true when running this in lab

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
speed_linear = 200                          # Set linear speed in mm/s
speed_joint = 150                           # joint speed in deg/s
accel_linear = 200                         # linear accel in mm/ss
accel_joints= 360                            # joint accel in deg/ss
robot.setSpeed(speed_linear,speed_joint,accel_linear,accel_joints)    # Set linear and joint speed 

# gripper doesn't work on rdk simulation, so no need to call it when simulating
if RUN_ON_ROBOT: 
    # ## Initialize Gripper
    instrument = minimalmodbus.Instrument(GRIPPER_PORT, 9, debug = False)
    instrument.serial.baudrate = 115200
    gripper = RobotiqGripper(portname=GRIPPER_PORT,slaveaddress=9)


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

# # dummy homogeneous transformation matrix to bring camera to aruco marker
# dummy_aruco_pose_correction = Mat(np.eye(4).tolist())
# # Generic identity matrix converted to RDK Pose
# # Note: Mat() only accepts nested lists, not np arrays     

# # corrected pose after aruco pose estimation
# corrected_bin_reference = bin_reference*dummy_aruco_pose_correction

# # pose above bins at which to detect and sort marker ids of the bins
# above_bins_pose = corrected_bin_reference*transl(above_bins_offset)

# # pose of bin 0 wrt bin reference
# bin_0_pose = corrected_bin_reference*transl(bin_relative_translations[0])