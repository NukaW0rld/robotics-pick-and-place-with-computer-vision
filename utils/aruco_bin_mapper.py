import cv2
import numpy as np
from cv2 import aruco
from robodk.robolink import *    # RoboDK API
from robodk.robomath import *    # Robot toolbox

def map_markers_to_bins(image_path, marker_ids, bin_locations):
    # load image and grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    
    # find markers
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(image)
    
    # Error handling if no markers found
    if ids is None:
        print("No markers found")
        return {}

    marker_to_bin = {}
    marker_positions = []
    
    # make ids flat
    ids = ids.flatten()
    
    # loop through markers
    for i, marker_id in enumerate(ids):
        # get center
        marker_corners = corners[i][0]
        center_x = np.mean([corner[0] for corner in marker_corners])
        center_y = np.mean([corner[1] for corner in marker_corners])
        
        # add to list
        marker_positions.append((int(marker_id), (center_x, center_y)))
    
    # sort by y then x
    sorted_markers = sorted(marker_positions, key=lambda m: (m[1][1], m[1][0]))
    
    # map markers to bins
    for i, (marker_id, _) in enumerate(sorted_markers):
        if marker_id in marker_ids and i < len(bin_locations):
            marker_to_bin[int(marker_id)] = bin_locations[i]
    
    return marker_to_bin

def estimate_marker_pose(frame, marker_id, marker_length=50):
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Choose ArUco dictionary
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    
    # Camera calibration parameters
    cameraMatrix = np.array([
        [666.1587209051977, 0.0, 314.3743381031827],
        [0.0, 663.0056540578997, 242.4330640690749],
        [0.0, 0.0, 1.0]
    ])
    
    distCoeffs = np.array([
        0.1767512926748398, -1.112191131510755,
        -0.002298693764898058, -0.006839192530929239,
        2.585175488403379
    ])
    
    # Detect ArUco markers
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(gray)
    
    # Check if any markers were detected
    if ids is not None:
        # Look for the specific marker ID
        ids = ids.flatten()
        for i, id in enumerate(ids):
            if id == marker_id:
                # Estimate pose of the marker
                # Convert marker_length from mm to meters for estimatePoseSingleMarkers
                marker_length_m = marker_length / 1000.0
                
                # Reshape corners for estimatePoseSingleMarkers
                corners_for_pose = [corners[i]]
                
                # Estimate pose
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                    corners_for_pose, marker_length_m, cameraMatrix, distCoeffs
                )
                
                # Extract single rotation and translation vectors
                rotation_vector = rvec[0][0]
                translation_vector = tvec[0][0]
                
                # Convert rotation vector to rotation matrix using Rodrigues formula
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                
                # Create 4x4 pose matrix (homogeneous transformation matrix)
                pose_matrix = np.eye(4)
                pose_matrix[:3, :3] = rotation_matrix
                pose_matrix[:3, 3] = translation_vector
                
                return pose_matrix
    
    # If marker not found
    print(f"Marker ID {marker_id} not found in the frame")
    return None

def compute_robot_bin_poses(frame, reference_marker_id, bin_marker_ids, bin_locations):
    # Step 1: Get the pose of the reference marker
    marker_7_pose = estimate_marker_pose(frame, reference_marker_id)
    if marker_7_pose is None:
        print(f"Reference marker {reference_marker_id} not found")
        return {}
    
    print("Step 1: Reference Marker Pose Matrix")
    print(marker_7_pose)
    print()
    
    # Step 2: Convert to RoboDK Mat and calculate the corrected reference pose
    marker_7_pose_mat = Mat(marker_7_pose.tolist())
    aruco_correction = invH(marker_7_pose_mat.translationPose())
    
    # Define the reference pose at which the ArUco marker is imaged
    reference_pose = Pose(-425.0, -300.0, 400, -180, 0, 180)
    
    # Calculate the corrected reference pose
    corrected_reference_pose = reference_pose * aruco_correction
    
    print("Step 2: Corrected Reference Pose")
    print(corrected_reference_pose)
    print()
    
    # Step 3: Calculate the pose for each bin
    bin_poses = {}
    print("Step 3: Bin Poses")
    
    for marker_id in bin_marker_ids:
        if marker_id in bin_locations:
            # Get the relative translation for this bin
            bin_location = bin_locations[marker_id]
            
            # Calculate the bin pose using the corrected reference pose and relative translation
            bin_pose = corrected_reference_pose * transl(bin_location)
            
            # Store the bin pose
            bin_poses[marker_id] = bin_pose
            
            # Print the bin pose
            print(f"Bin {marker_id} Pose:")
            print(bin_pose)
            print()
    
    return bin_poses
