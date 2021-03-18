import numpy as np
import os
import sys
# not a good way, but for now suffices
sys.path.append('/media/dtejaswi/DATA/CommonData/Projects/student_projects/joe_daniel/code')

import argparse
import utils
import matplotlib.pyplot as plt


def process_grountruth_relative_poses(sequence_dir, force_recalculate=False):
    """
    Process a directory of '000000x.txt' absolute poses to a numpy array of relative poses.
    """

    file_base_endeff = os.path.join(sequence_dir, "poses_gt_base_endeff.npy")
    file_first_endeff = os.path.join(sequence_dir, "poses_gt_first_endeff.npy")
    file_prev_curr_endeff = os.path.join(sequence_dir, "poses_gt_prev_curr_endeff.npy")

    file_base_cam = os.path.join(sequence_dir, "poses_gt_base_cam.npy")
    file_first_cam = os.path.join(sequence_dir, "poses_gt_first_cam.npy")
    file_prev_curr_cam = os.path.join(sequence_dir, "poses_gt_prev_curr_cam.npy")

    # safety check to not overwrite existing poses
    if os.path.exists(file_base_endeff) and not force_recalculate:
        print("Using existing pose array saved at {}".format(file_base_endeff))
        return (np.load(file_base_endeff), np.load(file_first_endeff), np.load(file_prev_curr_endeff),
                np.load(file_base_cam), np.load(file_first_cam), np.load(file_prev_curr_cam))

    poses_base_endeff = np.empty((1, 4, 4))
    poses_first_endeff = np.empty((1, 4, 4))
    poses_prev_curr_endeff = np.empty((1, 4, 4))
    poses_base_cam = np.empty((1, 4, 4))
    poses_first_cam = np.empty((1, 4, 4))
    poses_prev_curr_cam = np.empty((1, 4, 4))

    # make a list of all the pose files in the folder and sort numerically
    pose_files = [f for f in os.listdir(sequence_dir) if f.endswith(".txt")]
    pose_files = [os.path.join(sequence_dir, f) for f in sorted(pose_files)]
    previous_pose_of_endeff = None
    previous_pose_of_cam = None
    first_pose_of_endeff = None
    first_pose_of_cam = None

    # transformation to convert from end effector to camera frame
    t_endeff_cam = np.eye(4)
    t_endeff_cam[0, 0] = -1
    t_endeff_cam[1, 1] = -1

    first_index = True
    for p in pose_files:
        # print(p)
        f = open(p, 'r')
        raw_pose = f.readlines()[0]
        raw_pose = raw_pose.replace('(', '').replace(')', '').replace(']', '').replace('[', '')
        raw_pose = np.fromstring(raw_pose, sep=',')
        
        if first_pose_of_endeff is None:
            first_pose_of_endeff = raw_pose
        # pose of end effector wrt base is the raw pose
        p_base_endeff = utils.get_4x4_from_pose(raw_pose, rotation_mode='axang')
        # pose of end effector wrt first pose
        p_first_endeff = utils.get_relative_6dof(first_pose_of_endeff[0:3], first_pose_of_endeff[3:],
                                                 raw_pose[0:3], raw_pose[3:], rotation_mode='axang',
                                                 return_as_mat=True)

        # pose of camera wrt base
        p_base_cam = p_base_endeff @ t_endeff_cam
        if first_pose_of_cam is None:
            first_pose_of_cam = p_base_cam  # this is a 4x4 matrix
        # pose of camera wrt first pose
        p_first_cam = utils.get_relative_6dof(first_pose_of_cam[0:3, 3], first_pose_of_cam[0:3, 0:3],
                                              p_base_cam[0:3, 3], p_base_cam[0:3, 0:3], rotation_mode='rotm',
                                              return_as_mat=True)

        # current end effector pose relative to previous end effector pose 
        if previous_pose_of_endeff is None:
            p_prev_curr_endeff = utils.get_relative_6dof(first_pose_of_endeff[0:3], first_pose_of_endeff[3:],
                                                         raw_pose[0:3], raw_pose[3:], rotation_mode='axang',
                                                         return_as_mat=True)
        else:
            p_prev_curr_endeff = utils.get_relative_6dof(previous_pose_of_endeff[0:3], previous_pose_of_endeff[3:],
                                                         raw_pose[0:3], raw_pose[3:],
                                                         rotation_mode='axang', return_as_mat=True)
        previous_pose_of_endeff = raw_pose  # this is in axang

        # current camera pose relative to previous camera pose
        if previous_pose_of_cam is None:
            p_prev_curr_cam = utils.get_relative_6dof(first_pose_of_cam[0:3, 3], first_pose_of_cam[0:3, 0:3],
                                                      p_base_cam[0:3, 3], p_base_cam[0:3, 0:3], rotation_mode='rotm',
                                                      return_as_mat=True)
        else:
            p_prev_curr_cam = utils.get_relative_6dof(previous_pose_of_cam[0:3, 3], previous_pose_of_cam[0:3, 0:3],
                                                      p_base_cam[0:3, 3], p_base_cam[0:3, 0:3], rotation_mode='rotm',
                                                      return_as_mat=True)
        previous_pose_of_cam = p_base_cam   # this is a 4x4 matrix

        if first_index:
            poses_base_endeff = p_base_endeff.reshape((1, 4, 4))
            poses_first_endeff = p_first_endeff.reshape((1, 4, 4))
            poses_prev_curr_endeff = p_prev_curr_endeff.reshape((1, 4, 4))

            poses_base_cam = p_base_cam.reshape((1, 4, 4))
            poses_first_cam = p_first_cam.reshape((1, 4, 4))
            poses_prev_curr_cam = p_prev_curr_cam.reshape((1, 4, 4))

            first_index = False
        else:
            poses_base_endeff = np.append(poses_base_endeff, p_base_endeff.reshape((1, 4, 4)), axis=0)
            poses_first_endeff = np.append(poses_first_endeff, p_first_endeff.reshape((1, 4, 4)), axis=0)
            poses_prev_curr_endeff = np.append(poses_prev_curr_endeff, p_prev_curr_endeff.reshape((1, 4, 4)), axis=0)

            poses_base_cam = np.append(poses_base_cam, p_base_cam.reshape((1, 4, 4)), axis=0)
            poses_first_cam = np.append(poses_first_cam, p_first_cam.reshape((1, 4, 4)), axis=0)
            poses_prev_curr_cam = np.append(poses_prev_curr_cam, p_prev_curr_cam.reshape((1, 4, 4)), axis=0)

    np.save(file_base_endeff, poses_base_endeff)
    np.save(file_first_endeff, poses_first_endeff)
    np.save(file_prev_curr_endeff, poses_prev_curr_endeff)

    np.save(file_base_cam, poses_base_cam)
    np.save(file_first_cam, poses_first_cam)
    np.save(file_prev_curr_cam, poses_prev_curr_cam)
    
    return (poses_base_endeff, poses_first_endeff, poses_prev_curr_endeff,
            poses_base_cam, poses_first_cam, poses_prev_curr_cam)


def visualise_ground_truth_trajectories(poses, title="", savename=None):
    fig = plt.figure()
    plt.suptitle(title)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=88, azim=-90)
    ax.plot3D(poses[:, 0, 3], poses[:, 1, 3], poses[:, 2, 3], 'k')
    ax.plot3D(poses[0, 0, 3], poses[0, 1, 3], poses[0, 2, 3], 'go')
    ax.plot3D(poses[-1, 0, 3], poses[-1, 1, 3], poses[-1, 2, 3], 'rx')

    for idx in range(0, poses.shape[0], 2):
        utils.draw_axes_at(poses[idx, :], ax, scale=0.03, frame_label=idx, arrowstyle='->')  # end effector wrt base

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim([-0.2, 0.6])
    ax.set_ylim([0, 0.8])
    ax.set_zlim([0, 0.8])
    if savename:
        fig.savefig(savename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process pose files")

    parser.add_argument("input_folder", type=str, default=None,
                        help="Folder where the pose and epirect files are present.")
    args = parser.parse_args()

    input_folder = args.input_folder
    print("input folder: {}".format(input_folder))
    ps_base_endeff, ps_first_endeff, ps_prev_curr_endeff,\
        ps_base_cam, ps_first_cam, ps_prev_curr_cam = process_grountruth_relative_poses(input_folder,
                                                                                        force_recalculate=True)

    # sanity checks
    # 1. convert from relative poses to absolute poses wrt first
    ps_first_endeff_from_relative = utils.rel_to_abs(ps_prev_curr_endeff)
    ps_first_cam_from_relative = utils.rel_to_abs(ps_prev_curr_cam)
    # 2. convert from poses in step 1 to absolute poses wrt base
    ps_base_endeff_from_relative = np.zeros((ps_first_endeff_from_relative.shape[0], 4, 4), dtype=np.float)
    for i in range(0, ps_first_endeff_from_relative.shape[0]):
        ps_base_endeff_from_relative[i, :, :] = ps_base_endeff[0, :, :] @ ps_first_endeff_from_relative[i, :, :]

    ps_base_cam_from_relative = np.zeros((ps_first_cam_from_relative.shape[0], 4, 4), dtype=np.float)
    for i in range(0, ps_base_cam_from_relative.shape[0]):
        ps_base_cam_from_relative[i, :, :] = ps_base_cam[0, :, :] @ ps_first_cam_from_relative[i, :, :]

    # visualise_ground_truth_trajectories(poses_prev_curr, savename=os.path.join(input_folder, "trajectory.png"))

    visualise_ground_truth_trajectories(ps_base_endeff, "ps_base_endeff", os.path.join(input_folder, "trajectory.png"))
    # visualise_ground_truth_trajectories(ps_base_cam, "ps_base_cam", None) # x, y should be flipped from above
    #
    # visualise_ground_truth_trajectories(ps_first_endeff, "ps_first_endeff", None)
    # # This should be the same as the ps_first_endeff
    # visualise_ground_truth_trajectories(ps_first_endeff_from_relative, "ps_first_endeff_from_relative", None)
    #
    # visualise_ground_truth_trajectories(ps_first_cam, "ps_first_cam", None)
    # # This should be the same as the ps_first_cam
    # visualise_ground_truth_trajectories(ps_first_cam_from_relative, "ps_first_cam_from_relative", None)
    #
    # # This should be the same as the ps_base_endeff
    # visualise_ground_truth_trajectories(ps_base_endeff_from_relative, "ps_base_endeff_from_relative", None)
    # # This should be the same as the ps_base_cam
    # visualise_ground_truth_trajectories(ps_base_cam_from_relative, "ps_base_cam_from_relative", None)
    #
    # err = ps_base_endeff - ps_base_endeff_from_relative
    # print(np.sum(err))    # this should be close to zero
    #
    # err = ps_base_cam - ps_base_cam_from_relative
    # print(np.sum(err))    # this should be close to zero
    # plt.show()
