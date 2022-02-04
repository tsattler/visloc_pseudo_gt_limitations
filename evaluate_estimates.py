import argparse
import numpy as np
import math
import json
import os
import random

import matplotlib.pyplot as plt

import evaluation_util

parser = argparse.ArgumentParser(
    description='Compute and plot errors of estimated poses wrt to pseudo ground truth.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data_config',
                    help='file containing the test case specification including paths to pseudo ground truth and estimates of algorithms')

parser.add_argument('--error_threshold', type=float, default=5,
                    help='Error threshold when calculating recall, and bound for plotting error curves.')

parser.add_argument('--error_type', type=str, default='pose', choices=['pose', 'dcre_max', 'dcre_mean'],
                    help='Choice of error type.')

parser.add_argument('--error_max_images', type=int, default=1000,
                    help='Use at most x images when calculating error distribution for speed. -1 for using all.')

opt = parser.parse_args()

sub_plot_width = 5
sub_plot_height = 4
sub_plot_y_label = 'percentage of frames'

if opt.error_type == 'pose':
    sub_plot_x_label = 'max. of deg/cm err.'
    plot_title = "Pose Error"
else:
    sub_plot_x_label = 're-proj. err. (px)'
    if opt.error_type == 'dcre_max':
        plot_title = "Max. DCRE"
    else:
        plot_title = "Mean DCRE"

plot_max_cols = 4
plot_out_file = f"{os.path.splitext(opt.data_config)[0]}_{opt.error_type}_err.pdf"

with open(opt.data_config, "r") as f:
    test_data = json.load(f)

plot_num = len(test_data["scenes"]) + 1 # scene plots + average plot
plot_cols = min(plot_num, plot_max_cols)
plot_rows = int(math.ceil(plot_num/plot_cols))

plt.figure(figsize=[sub_plot_width*plot_cols,sub_plot_height*plot_rows])
plt.suptitle(plot_title)

algo_metrics = {}
dataset_folder = test_data["folder"]
rgb_image_width = test_data['image_width']

for s_idx, scene_data in enumerate(test_data["scenes"]):

    scene_name = scene_data['name']
    scene_folder = scene_data['folder']

    print(f"\n{scene_name}")
    plt.subplot(plot_rows, plot_cols, 2+s_idx) # reserve first sub plot for average
    plt.title(scene_name)

    # load pseudo ground truth
    pgt_poses = evaluation_util.read_pose_data(scene_data["pgt"])

    if opt.error_max_images > 0 and len(pgt_poses) > opt.error_max_images:
        keys = random.sample(pgt_poses.keys(), opt.error_max_images)
        pgt_poses = {k: pgt_poses[k] for k in keys}

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
                pgt_pose, rgb_focal_length = pgt_poses[query_file]
                est_pose, _ = est_poses[query_file]

                if opt.error_type=='pose':
                    errors[i] = evaluation_util.compute_error_max_rot_trans(pgt_pose, est_pose)
                else:
                    depth_file = evaluation_util.infer_depth_file_from_image_file(query_file)
                    errors[i] = evaluation_util.compute_error_dcre(
                        os.path.join(dataset_folder, scene_folder, depth_file),
                        pgt_pose, est_pose,
                        rgb_focal_length, rgb_image_width,
                        use_max=(opt.error_type=='dcre_max'))

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
