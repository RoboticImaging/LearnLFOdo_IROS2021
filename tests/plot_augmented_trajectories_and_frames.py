# This script plots the ground truth trajectories as recorded by the robot arm and the coordinate frames
# of the as it moves along the trajectory. Do not mess with rotations in this code ever!!!!
#
# Author and Maintainer: Tejaswi Digumarti (tejaswi.digumarti@sydney.edu.au)

import numpy as np
import utils
import os

import matplotlib.pyplot as plt


if __name__ == "__main__":
    sequence = 2
    folder = "/media/dtejaswi/DATA/CommonData/Projects/student_projects/joe_daniel/data/module-1-1/module1-3-png/"

    poses_base_endeff = np.load(os.path.join(folder, "seq"+str(sequence), "poses_gt_base_endeff.npy"))
    poses_base_cam = np.load(os.path.join(folder, "seq"+str(sequence), "poses_gt_base_cam.npy"))

    poses_base_cam_rot_flip = np.load(os.path.join(folder, "seq"+str(sequence+700), "poses_gt_base_cam.npy"))

    # -------------------
    # plot trajectories
    # -------------------
    fig = plt.figure()
    plt.suptitle(str(sequence))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=88, azim=-90)

    # plot x, y, z positions
    ax.plot3D(poses_base_endeff[:, 0, 3], poses_base_endeff[:, 1, 3], poses_base_endeff[:, 2, 3], 'k', linewidth=0.3)
    ax.plot3D(poses_base_cam[:, 0, 3], poses_base_cam[:, 1, 3], poses_base_cam[:, 2, 3], 'r', linewidth=0.3)
    ax.plot3D(poses_base_cam_rot_flip[:, 0, 3], poses_base_cam_rot_flip[:, 1, 3], poses_base_cam_rot_flip[:, 2, 3], 'k', linewidth=0.3)


    # draw coordinate axes of the end effector and the camera in the base frame
    for i in range(0, poses_base_endeff.shape[0], 5):
    # for i in range(120, 200, 2):
        # utils.draw_axes_at(poses_base_endeff[i, :, :], ax, scale=0.01, frame_label=i, arrowstyle='->')       # end effector wrt base
        # utils.draw_axes_at(poses_base_cam[i, :, :], ax, scale=0.03, frame_label=i, arrowstyle='->')   # camera wrt base
        utils.draw_axes_at(poses_base_cam_rot_flip[i, :, :], ax, scale=0.03, frame_label=i, arrowstyle='->')  # camera wrt base

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.set_xlim([-0.8, 0.8])
    ax.set_ylim([0, 0.8])
    ax.set_zlim([0, 0.8])
    plt.show()
