# This script converts trajectories predicted by the different approaches into
# the TUM RGBD-dataset trajectory format
# (https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats)
# timestamp tx ty tz qx qy qz qw
#
# Author and Maintainer: Tejaswi Digumarti (tejaswi.digumarti@sydney.edu.au)

import numpy as np
import os
import math
import utils
# from  tf.transformations import quaternion_from_euler

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    """Return quaternion from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    >>> q = quaternion_from_euler(1, 2, 3, 'ryxz')
    >>> np.allclose(q, [0.310622, -0.718287, 0.444435, 0.435953])
    True

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    quaternion = np.empty((4, ), dtype=np.float64)
    if repetition:
        quaternion[i] = cj*(cs + sc)
        quaternion[j] = sj*(cc + ss)
        quaternion[k] = sj*(cs - sc)
        quaternion[3] = cj*(cc - ss)
    else:
        quaternion[i] = cj*sc - sj*cs
        quaternion[j] = cj*ss + sj*cc
        quaternion[k] = cj*cs - sj*sc
        quaternion[3] = cj*cc + sj*ss
    if parity:
        quaternion[j] *= -1

    return quaternion


def rel_to_abs(poses, global_transform=None):
    poses_abs = np.zeros((poses.shape[0]+1, poses.shape[1]), dtype=np.float)

    poses_abs[0,:] = np.zeros((1,6))

    prev_tr = np.eye(4)

    if global_transform is None:
        global_transform = np.eye(4)

    for i in range(poses.shape[0]):
        t = poses[i,0:3].reshape(1,3)   # translation

        # convert each relative pose from [x, y, z, euler angles] to 4x4 transformation matrix
        rotm = utils.euler_to_rotm(poses[i, 3], poses[i, 4], poses[i, 5])
        tr = np.concatenate((rotm, np.transpose(t)), axis=1)
        tr = np.concatenate((tr, np.zeros((1, 4))), axis=0)
        tr[3, 3] = 1

        # apply the transformation from start upto previous pose to the current relative pose
        prev_tr = tr @ prev_tr
        final_tr = prev_tr @ global_transform
        # print("prev: {} final: {}".format(prev_tr, final_tr))
        # convert back to [x, y, z, euler angles]
        poses_abs[i+1,0:3] = final_tr[0:3,3].transpose()
        poses_abs[i+1, 3:] = utils.rotm_to_euler(final_tr[0:3, 0:3])

    return poses_abs

def convert_pose_to_tum(poses_abs, name):
    tum_pose = []
    for i in range(poses_abs.shape[0]):
        t = poses_abs[i, 0:3].reshape(3,)
        rpy = poses_abs[i, 3:].reshape(3,)
        # print(rpy)
        quat = quaternion_from_euler(rpy[0], rpy[1], rpy[2], 'rxyz')

        tum_pose.append([i, t[0], t[1], t[2], quat[0], quat[1], quat[2], quat[3]])
    np.savetxt(name, tum_pose, delimiter=" ")


def main():

    data_dir = "/media/dtejaswi/DATA/CommonData/Projects/student_projects/joe_daniel/data/module-1-1/module1-2-png/"
    # results_dir = "/home/dtejaswi/tensorboard_hpc/artemis_test_b16_interarea/"

    results_dir = "/media/dtejaswi/tejaswi1/joe_artemis_b16_copy/artemis_test_b16_interarea_tv_mean/"

    sequences = [16, 28, 40, 44, 60, 61, 62]
    # sequences = [16]
    epoch = 40

    for seq in sequences:
        print(seq)
        input_dir = os.path.join(data_dir, "seq"+str(seq))
        # ground truth poses each pose relative to its previous pose
        pose_rel_gt = np.load(os.path.join(input_dir, "poses_gt_relative.npy"))
        # ground truth poses relative to the first frame
        pose_abs_gt = np.load(os.path.join(input_dir, "poses_gt_absolute.npy"))

        # convert absolute ground truth poses to TUM format
        convert_pose_to_tum(pose_abs_gt, os.path.join(input_dir, "pose_abs_gt_tum.txt"))

        modes = ["singlewarp", "multiwarp-5"]
        for mode in modes:
            if mode == "singlewarp":
                encodings = ["monocular", "stack", "focalstack-17-5", "focalstack-17-9", "epi", "epi_without_disp_stack"]
            else:
                encodings = ["stack", "focalstack-17-5", "focalstack-17-9", "epi", "epi_without_disp_stack"]

            for enc in encodings:
                # estimated poses are relative to the previous pose
                estimated_pose_file = os.path.join(results_dir, mode, enc, "results", "seq"+str(seq)+"_epoch_"+str(epoch), "poses.npy")
                pose_rel_est = np.load(estimated_pose_file)
                # pose_rel_est = np.loadtxt(estimated_pose_file)

                # convert them relative to the first frame
                pose_abs_est = rel_to_abs(pose_rel_est)

                # convert the result to TUM format
                out_pose_file = os.path.join(results_dir, mode, enc, "results", "seq"+str(seq)+"_epoch_"+str(epoch), "pose_abs_est_tum.txt")
                convert_pose_to_tum(pose_abs_est[1:], out_pose_file)

                # pose_abs_est[:, 0] = - pose_abs_est_[:, 2]
                # pose_abs_est[:, 2] = - pose_abs_est_[:, 0]
                # pose_abs_est[:, 3] = pose_abs_est_[:, 3] - 45
                # # pose_abs_est[:, 4] = pose_abs_est_[:, 4] + 45
                # # pose_abs_est[:, 5] = pose_abs_est_[:, 5] - 45

                # convert the result to TUM format
                out_pose_file = os.path.join(results_dir, mode, enc, "results",
                                             "seq" + str(seq) + "_epoch_" + str(epoch), "pose_abs_est_tr_tum.txt")
                convert_pose_to_tum(pose_abs_est, out_pose_file)

if __name__ == "__main__":
    main()