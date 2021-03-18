# This script plots the ground truth trajectories as recorded by the robot arm and the coordinate frames
# of the as it moves along the trajectory. Do not mess with rotations in this code ever!!!!
#
# Author and Maintainer: Tejaswi Digumarti (tejaswi.digumarti@sydney.edu.au)

import numpy as np
import utils
import os

import matplotlib.pyplot as plt


if __name__ == "__main__":
    sequence = 14
    folder = "/media/dtejaswi/Seagate Expansion Drive/JoeDanielThesisData/data/sequences/seq" + str(sequence)
    # make a list of all the pose files in the folder and sort numerically
    pose_files = [f for f in os.listdir(folder) if f.endswith(".txt")]
    pose_files = [os.path.join(folder, f) for f in sorted(pose_files)]

    # read poses and load into an array
    poses = []
    for i in range(0, len(pose_files)):
        pose_file_name = '{0:06d}.txt'.format(i)
        file_path = open(os.path.join(folder, pose_file_name), 'r')
        p = file_path.readlines()[0]
        p = p.replace('(', '').replace(')', '').replace(']', '').replace('[', '')
        p = np.fromstring(p, sep=',')

        poses.append(p)
    poses = np.array(poses).reshape((-1, 6))

    # transformation to convert from end effector to camera frame
    transform_e_c = np.eye(4)
    transform_e_c[0, 0] = -1
    transform_e_c[1, 1] = -1

    # any other transformation that we wish to apply
    additional_transform = np.eye(4)
    additional_transform[0, 0] = additional_transform[1, 1] = 0
    additional_transform[0, 1] = -1
    additional_transform[1, 0] = 1

    poses_w_c = np.zeros((poses.shape[0], 4, 4))        # keep the altered poses as transformation matrices
    for i in range(poses.shape[0]):
        pose_w_e = utils.get_4x4_from_pose(poses[i, :])
        pose_w_c = pose_w_e @ transform_e_c @ additional_transform  # this chaining of transformations is correct
        poses_w_c[i, :, :] = pose_w_c

    # -------------------
    # plot trajectories
    # -------------------
    fig = plt.figure()
    plt.suptitle(str(sequence))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=88, azim=-90)

    # plot x, y, z positions
    ax.plot3D(poses[:, 0], poses[:, 1], poses[:, 2], 'k')
    ax.plot3D(poses[0, 0], poses[0, 1], poses[0, 2], 'go')
    ax.plot3D(poses[-1, 0], poses[-1, 1], poses[-1, 2], 'rx')

    # ax.plot3D(poses_w_c[:, 0, 3], poses_w_c[:, 1, 3], poses_w_c[:, 2, 3], 'b')
    # ax.plot3D(poses_w_c[0, 0, 3], poses_w_c[0, 1, 3], poses_w_c[0, 2, 3], 'go')
    # ax.plot3D(poses_w_c[-1, 0, 3], poses_w_c[-1, 1, 3], poses_w_c[-1, 2, 3], 'rx')

    # draw coordinate axes of the end effector and the camera in the base frame
    for i in range(0, len(poses), 2):
        pose_w_e = utils.get_4x4_from_pose(poses[i, :])
        utils.draw_axes_at(pose_w_e[:], ax, scale=0.01, frame_label=i, arrowstyle='->')       # end effector wrt base
        utils.draw_axes_at(poses_w_c[i, :], ax, scale=0.03, frame_label=i, arrowstyle='->')   # camera wrt base

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.set_xlim([-0.8, 0.8])
    ax.set_ylim([0, 0.8])
    ax.set_zlim([0, 0.8])
    plt.show()
