import argparse
from skimage import io
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rotation
import warnings
import cv2 as cv
import time
import math
import json
import os

import matplotlib.pyplot as plt

import evaluation_util

parser = argparse.ArgumentParser(
    description='Compute and plot errors of estimated poses wrt to pseudo ground truth.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data_config', help='file containing the test case specification including paths to pseudo ground truth and estimates of algorithms')

parser.add_argument('--error_threshold', type=float, default=5, help='Error threshold when calculating recall, and bound for plotting error curves.')

opt = parser.parse_args()

sub_plot_width = 5
sub_plot_height = 4
sub_plot_x_label = 'max. of deg/cm err.'
sub_plot_y_label = 'percentage of frames'

plot_max_cols = 4
plot_out_file = f"{os.path.splitext(opt.data_config)[0]}.pdf"

with open(opt.data_config, "r") as f:
    test_data = json.load(f)

plot_num = len(test_data["scenes"]) + 1 # scene plots + average plot
plot_cols = min(plot_num, plot_max_cols)
plot_rows = int(math.ceil(plot_num/plot_cols))

plt.figure(figsize=[sub_plot_width*plot_cols,sub_plot_height*plot_rows])
plt.suptitle("Pose Error")

algo_metrics = {}

for s_idx, scene_data in enumerate(test_data["scenes"]):

    print(f"\n{scene_data['name']}")
    plt.subplot(plot_rows, plot_cols, 2+s_idx) # reserve first sub plot for average
    plt.title(scene_data['name'])

    # load pseudo ground truth
    pgt_poses = evaluation_util.read_pose_data(scene_data["pgt"])

    # iterate through algorithm estimates
    for estimate_data in scene_data["estimates"]:

        # initialise algorithm accumulated metrics
        if estimate_data['algo'] not in algo_metrics:
            algo_metrics[estimate_data['algo']] = []

        # load estimated poses
        est_poses = evaluation_util.read_pose_data(estimate_data["estimated_poses"])

        # main evaluation loop
        errors = np.ndarray((len(pgt_poses), ))

        for i, query_file in enumerate(pgt_poses):

            try:
                errors[i] = evaluation_util.compute_error_max_rot_trans(
                    pgt_poses[query_file],
                    est_poses[query_file])
            except KeyError:
                # catching the case that an algorithm did not provide an estimate
                errors[i] = math.inf

        recall =  np.mean(errors < opt.error_threshold)
        hist_errors, cum_recall_base = np.histogram(errors, bins=100, range=[0,opt.error_threshold])

        cum_recall = np.cumsum(hist_errors) / errors.shape[0]
        cum_recall_base = cum_recall_base[:-1]

        print(f"\t{estimate_data['algo']:15s} {recall*100:5.1f}%")

        plt.plot(cum_recall_base, cum_recall, label=estimate_data['algo'])
        plt.xlabel(sub_plot_x_label)
        plt.ylabel(sub_plot_y_label)
        plt.legend(loc="best")

        algo_metrics[estimate_data['algo']].append((recall,cum_recall,cum_recall_base))

print(f"\nAverage")
plt.subplot(plot_rows, plot_cols, 1)
plt.title("Average")

for algo, metrics in algo_metrics.items():

    # average recall
    recalls = [m[0] for m in metrics]
    avg_recall = sum(recalls) / len(recalls)

    # average recall curves
    cum_recalls = [m[1] for m in metrics]
    avg_cum_recall = np.mean(cum_recalls, axis=0)

    avg_cum_recall_base = metrics[0][2] # plot bases should be the same, take first one

    print(f"\t{algo:15} {avg_recall * 100:5.1f}%")

    plt.plot(avg_cum_recall_base, avg_cum_recall, label=algo)
    plt.xlabel(sub_plot_x_label)
    plt.ylabel(sub_plot_y_label)
    plt.legend(loc="best")

plt.tight_layout()
plt.savefig(plot_out_file)

exit()


#TODO: check this:
#   we ignore estimates where no ground truth exists
#   we do not check whether an estimate is missing
#   do all methods provide complete estimates? should we check for that?
#   --> solution: cycle through ground truth, store inf error for missing estimates?


print(pgt_poses[0])
print(est_poses[0])

exit()

file = open(estimated_poses, "r")
pairs = file.readlines()

print('Checking estimated for valid ground truth.')
checked_pairs = []

for pair in pairs:

    query_name = pair.split()[0]

    if file_format == 1:
        query_name = query_name[:6] + scene_delimiter + query_name[7:]
    elif file_format == 2:
        query_name = query_name[5:]

    pose_file = query_name.split(".color")[0] + '.pose.txt'

    try:
        pose_gt = np.loadtxt('../datasets/' + scene_folder + '/test/poses/' + pose_file)
    except:
        continue

    checked_pairs.append(pair)

print('Found %d invalid estiamtes.' % (len(pairs) - len(checked_pairs)))
pairs = checked_pairs

num_within_error_thresh = 0
counter = 0

results = np.zeros((len(pairs), 4))

# number_of_visualisations = min(number_of_visualisations, 1+int((len(pairs)-1) / vis_skip))
# pts_per_image = num_points / len(pairs)
# img_vis_data_list = []

pose_acc = 0

for i in tqdm(range(0, len(pairs))):
    query_name = pairs[i].split()[0]

    if file_format == 1:
        query_name = query_name[:6] + scene_delimiter + query_name[7:]
    elif file_format == 2:
        query_name = query_name[5:]

    pose_file = query_name.split(".color")[0] + '.pose.txt'
    pose_gt = np.loadtxt('../datasets/' + scene_folder + '/test/poses/' + pose_file)

    # calibration_file = query_name.split(".color")[0] + '.calibration.txt'
    # calibration = np.loadtxt('../datasets/' + scene_folder + '/test/calibration/' + calibration_file)
    # calibration = float(calibration)

    # depth_file = query_name.split(".color")[0] + ".depth.png"
    # depth = io.imread('../datasets/' + scene_folder + '/test/depth/' + depth_file)
    # depth = depth.astype(np.float64)
    # depth /= 1000 # from millimeters to meters

    # d_h = depth.shape[0]
    # d_w = depth.shape[1]

    # reproject depth map to 3D eye coordinates
    # eye_coords = np.zeros((4, d_h, d_w))
    # set x and y coordinates
    # eye_coords[0] = np.dstack([np.arange(0,d_w)] * d_h)[0].T
    # eye_coords[1] = np.dstack([np.arange(0,d_h)] * d_w)[0]

    # eye_coords = eye_coords.reshape(4, -1)
    # depth = depth.reshape(-1)

    # filter pixels with invalid depth
    # depth_mask = (depth > 0.3) & (depth < 5)
    # eye_coords = eye_coords[:, depth_mask]
    # depth = depth[depth_mask]

    # eye_coords = torch.from_numpy(eye_coords).cuda()
    # depth = torch.from_numpy(depth).cuda()

    # save original pixel positions for later
    # pixel_coords = eye_coords[0:2].clone()

    # substract depth principal point (assume image center)
    # eye_coords[0] -= d_w / 2
    # eye_coords[1] -= d_h / 2
    # reproject
    # eye_coords[0:2] *= depth / calibration
    # eye_coords[2] = depth
    # eye_coords[3] = 1

    # Obtains the transformation from the global into the local camera
    # coordinate system of the estimated pose. Assumes that the estimated
    # pose is given as q, t, where q is the rotation from global to local
    # coordinates and t is the translation.
    q = np.array(pairs[i].split()[1:5]).astype(np.float64)
    q_vec = np.array([q[1], q[2], q[3], q[0]])
    t = np.array(pairs[i].split()[5:8]).astype(np.float64)
    R = Rotation.from_quat(q_vec).as_matrix()

    pose_est = np.identity(4)
    pose_est[0:3, 0:3] = R
    pose_est[0:3, 3] = t

    pose_est_inv = torch.from_numpy(np.linalg.inv(pose_est))
    pose_est = torch.from_numpy(pose_est)
    pose_gt = torch.from_numpy(pose_gt)

    # calculate pose errors
    t_err = float(torch.norm(pose_gt[0:3, 3] - pose_est_inv[0:3, 3]))

    gt_R = pose_gt[0:3, 0:3].numpy()
    out_R = pose_est_inv[0:3, 0:3].numpy()

    r_err = np.matmul(out_R, np.transpose(gt_R))
    r_err = cv.Rodrigues(r_err)[0]
    r_err = np.linalg.norm(r_err) * 180 / math.pi

    results[i, 2] = float(r_err)
    results[i, 3] = float(t_err * 100)

    if r_err < 5 and t_err < 0.05:
        pose_acc += 1

    # pose_est = pose_est.cuda()
    # pose_gt = pose_gt.cuda()

    # scene_coords = torch.matmul(pose_gt, eye_coords)
    # eye_coords2 = torch.matmul(pose_est, scene_coords)

    # eye_coords = eye_coords[0:3]
    # eye_coords2 = eye_coords2[0:3]

    # dist_errors = torch.norm(eye_coords - eye_coords2, p=2, dim=0) * 100
    # dist_viz = np.zeros((d_h, d_w), dtype='uint8')

    # if file_format == 1:
    #    #active search for 7Scenes
    #    framenumber = int(query_name[13:-10])
    # else:
    #    framenumber = int(query_name[6:-10])

    # if number_of_visualisations > 0 and not framenumber % vis_skip:

    #    number_of_visualisations -= 1

    #    for pt in range(dist_errors.shape[0]):
    #        x = round(float(pixel_coords[0, pt]))
    #        y = round(float(pixel_coords[1, pt]))
    #        e = float(dist_errors[pt])

    #        dist_viz[y, x] = min(max(e / threshold_for_visualization * 255, 0), 255)

    #    dist_viz = cv.applyColorMap(dist_viz, cv.COLORMAP_INFERNO)

    #    with warnings.catch_warnings():
    #            warnings.simplefilter("ignore")
    #            io.imsave(output_reprojection_errors[:-4] + '_%d.png' % i, dist_viz)

    # results[i, 0] = float(torch.max(dist_errors))
    # results[i, 1] = float(torch.mean(dist_errors))

    # pt_skip = int(dist_errors.shape[0] / pts_per_image)
    # scene_coords = scene_coords[:, ::pt_skip]
    # dist_errors = dist_errors[::pt_skip]
    # depth = depth[::pt_skip]

    # img_vis_data = np.ndarray([depth.shape[0], 5], dtype=float)
    # img_vis_data[:,0:3] = scene_coords.cpu().numpy().transpose()[:,0:3]
    # img_vis_data[:,3] = dist_errors.cpu().numpy()
    # img_vis_data[:,4] = depth.cpu().numpy()
    # img_vis_data_list.append(img_vis_data)

# np.save(output_for_visualization, np.concatenate(img_vis_data_list, axis=0))
np.save(output_reprojection_errors, results)
print("5cm5deg: %.1f%%" % (pose_acc / len(pairs) * 100))
