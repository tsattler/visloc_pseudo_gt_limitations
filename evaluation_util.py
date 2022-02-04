import math
import os
import numpy as np
import cv2 as cv
import torch
from skimage import io

from scipy.spatial.transform import Rotation as Rotation

def read_pose_data(file_name):
    '''
    Expects path to file with one pose per line.
    Input pose is expected to map world to camera coordinates.
    Output pose maps camera to world coordinates.
    Pose format: file qw qx qy qz tx ty tz (f)
    Return dictionary that maps a file name to a tuple of (4x4 pose, focal_length)
    Sets focal_length to None if not contained in file.
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

        if len(pose_string) > 8:
            focal_length = float(pose_string[8])
        else:
            focal_length = None

        pose_dict[file_name] = (pose_4x4, focal_length)

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

def infer_depth_file_from_image_file(image_file):
    '''
    Map an image file the corresponding depth file.
    Assumes prior knowledge about the dataset structure, thus adapt for new datasets.
    '''

    # assumes image file format: some_name.color.xyz
    # assumes depth file format: some_name.depth.png

    depth_file, _ = os.path.splitext(image_file)
    depth_file, _ = os.path.splitext(depth_file)
    return f"{depth_file}.depth.png"

def compute_error_dcre(depth_file, pgt_pose, est_pose, rgb_focal_length, rgb_image_width, use_max=True):
    '''
    Compute the dense reprojection error.
    Expects poses to map camera coordinate to world coordinates.
    Needs access to image depth.
    Calculates the max. DCRE per images, or the mean if use_max=False.
    '''

    pgt_pose = torch.from_numpy(pgt_pose).cuda()
    est_pose = torch.from_numpy(est_pose).cuda()

    depth = io.imread(depth_file)
    depth = depth.astype(np.float64)
    depth /= 1000  # from millimeters to meters

    d_h = depth.shape[0]
    d_w = depth.shape[1]

    rgb_to_d_scale = d_w / rgb_image_width
    d_focal_length = rgb_focal_length * rgb_to_d_scale

    # reproject depth map to 3D eye coordinates
    prec_eye_coords = np.zeros((4, d_h, d_w))
    # set x and y coordinates
    prec_eye_coords[0] = np.dstack([np.arange(0, d_w)] * d_h)[0].T
    prec_eye_coords[1] = np.dstack([np.arange(0, d_h)] * d_w)[0]
    prec_eye_coords = prec_eye_coords.reshape(4, -1)

    eye_coords = prec_eye_coords.copy()
    depth = depth.reshape(-1)

    # filter pixels with invalid depth
    depth_mask = (depth > 0.3) & (depth < 10)
    eye_coords = eye_coords[:, depth_mask]
    depth = depth[depth_mask]

    eye_coords = torch.from_numpy(eye_coords).cuda()
    depth = torch.from_numpy(depth).cuda()

    # save original pixel positions for later
    pixel_coords = eye_coords[0:2].clone()

    # substract depth principal point (assume image center)
    eye_coords[0] -= d_w / 2
    eye_coords[1] -= d_h / 2
    # reproject
    eye_coords[0:2] *= depth / d_focal_length
    eye_coords[2] = depth
    eye_coords[3] = 1

    # transform to world and back to cam
    scene_coords = torch.matmul(pgt_pose, eye_coords)
    eye_coords = torch.matmul(torch.inverse(est_pose), scene_coords)

    # project
    depth = eye_coords[2]
    eye_coords = eye_coords[0:2]

    eye_coords *= (d_focal_length / depth)

    # add RGB principal point (assume image center)
    eye_coords[0] += d_w / 2
    eye_coords[1] += d_h / 2

    reprojection_errors = torch.norm(eye_coords - pixel_coords, p=2, dim=0)

    if use_max:
        return float(torch.max(reprojection_errors)) / rgb_to_d_scale
    else:
        return float(torch.mean(reprojection_errors)) / rgb_to_d_scale