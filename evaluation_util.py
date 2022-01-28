import math

import numpy as np
import cv2 as cv

from scipy.spatial.transform import Rotation as Rotation

def read_pose_data(file_name):
    '''
    Expects path to file with one pose per line.
    Pose is expected to map world coordinates to camera coordinates.
    Pose format: file qw qx qy qz tx ty tz
    '''

    with open(file_name, "r") as f:
        pose_data = f.readlines()

    # create a dict from the poses with file name as key
    pose_dict = {}
    for pose_string in pose_data:

        pose_string = pose_string.split()
        file_name = pose_string[0]

        pose_q = np.array(pose_string[1:5])
        pose_q = np.array([pose_q[1], pose_q[2], pose_q[3], pose_q[0]])
        pose_t = np.array(pose_string[5:8])
        pose_R = Rotation.from_quat(pose_q).as_matrix()

        pose_4x4 = np.identity(4)
        pose_4x4[0:3, 0:3] = pose_R
        pose_4x4[0:3, 3] = pose_t

        # convert world->cam to cam->world for evaluation
        pose_4x4 = np.linalg.inv(pose_4x4)

        pose_dict[file_name] = pose_4x4

    return pose_dict

def compute_error_max_rot_trans(pgt_pose, est_pose):
    '''
    Compute the pose error.
    Expects poses to map camera coordinate to world coordinates.
    '''

    # calculate pose errors
    t_err = float(np.linalg.norm(pgt_pose[0:3, 3] - est_pose[0:3, 3]))

    r_err = est_pose[0:3, 0:3] @ np.transpose(pgt_pose[0:3, 0:3])
    r_err = cv.Rodrigues(r_err)[0]
    r_err = np.linalg.norm(r_err) * 180 / math.pi

    return max(r_err, t_err*100)