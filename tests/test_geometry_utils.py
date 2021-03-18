# This script tests the geometry utils

import torch
import numpy as np
import utils

DEVICE = torch.device('cuda')


def test_axang_to_rotm():
    print("============================================================")
    print("Test axis angle to rotation matrix")
    print("============================================================")
    # define some rotation in axis angle notation with the magnitude representing the angle in radians
    axang_array = np.array([0.2341605, 0.468321, 0], dtype=np.float)
    # convert from axis angle to rotation matrrix
    rotm = utils.axang_to_rotm(axang_array, with_magnitude=True)

    # the expected answer is
    # [0.8928203,  0.0535898,  0.4472136;
    #  0.0535898,  0.9732051, -0.2236068;
    # -0.4472136,  0.2236068,  0.8660254]

    print(rotm.shape)
    print(rotm)
    print("============================================================")


def test_axang_to_rotm_tensor():
    print("============================================================")
    print("Test Euler ZYX to rotation matrix for tensors")
    print("============================================================")
    # define some rotation in axis angle notation with the magnitude representing the angle in radians
    axang_array = np.array([[0.2341605, 0.468321, 0],
                            [0.2285172, -0.914069, 0.4570345]], dtype=np.float)  # [2, 3] array. 2 is batch size
    axang_tensor = torch.from_numpy(axang_array).to(DEVICE)  # [2, 3] tensor, 2 is the batch size

    # convert from axis angle to rotation matrrix
    rotm_tensor = utils.axang_to_rotm_tensor(axang_tensor, with_magnitude=True)  # this should be a [2, 3, 3] tensor

    # the expected answer is
    # [0.8928203,  0.0535898,  0.4472136;   [0.5238096, -0.4732026, -0.7083099;
    #  0.0535898,  0.9732051, -0.2236068;    0.2827264,  0.8809524, -0.3794584;
    # -0.4472136,  0.2236068,  0.8660254]    0.8035480, -0.0014940,  0.5952381]

    print(rotm_tensor.size())
    print(rotm_tensor)


def test_euler_to_rotm():
    print("============================================================")
    print("Test Euler ZYX to rotation matrix")
    print("============================================================")
    rotm = utils.euler_to_rotm(0.2526803, 0.4636476, 0.0599512, rotation_order="zyx")
    # the expected answer is
    # [0.8928203,  0.0535898,  0.4472136;
    #  0.0535898,  0.9732051, -0.2236068;
    # -0.4472136,  0.2236068,  0.8660254]
    print(rotm)
    print("============================================================")


def test_euler_to_rotm_tensor():
    print("============================================================")
    print("Test Euler ZYX to rotation matrix for tensors")
    print("============================================================")
    euler_array = np.array([[0.2526803, 0.4636476, 0.0599512], [-0.0025099, -0.9332321, 0.49494]], dtype=np.float)
    euler_tensor = torch.from_numpy(euler_array).to(DEVICE)
    rotm = utils.euler_to_rotm_tensor(euler_tensor, rotation_order="zyx")
    # the expected answer is
    # [0.8928203,  0.0535898,  0.4472136;   [0.5238096, -0.4732026, -0.7083099;
    #  0.0535898,  0.9732051, -0.2236068;    0.2827264,  0.8809524, -0.3794584;
    # -0.4472136,  0.2236068,  0.8660254]    0.8035480, -0.0014940,  0.5952381]
    print(rotm.size())
    print(rotm)
    print("============================================================")


def test_quat_to_rotm():
    print("============================================================")
    print("Test quaternion to rotation matrix")
    print("============================================================")
    quat = np.array([0.8660254, 0.1091089, -0.4364358, 0.2182179])
    rotm = utils.quat_to_rotm(quat, order="wxyz")
    # The expected answer is
    # [0.5238096, -0.4732026, -0.7083099;
    #  0.2827264,  0.8809524, -0.3794584;
    #  0.8035480, -0.0014940,  0.5952381]
    print(rotm.shape)
    print(rotm)
    print("============================================================")


def test_quat_to_rotm_tensor():
    print("============================================================")
    print("Test quaternion to rotation matrix for tensors")
    print("============================================================")
    quat_array = np.array([[0.9659258, 0.1157474, 0.2314948, 0],
                           [0.8660254, 0.1091089, -0.4364358, 0.2182179]], dtype=np.float)
    quat_tensor = torch.from_numpy(quat_array).to(DEVICE)
    rotm = utils.quat_to_rotm_tensor(quat_tensor, order="wxyz")
    # the expected answer is
    # [0.8928203,  0.0535898,  0.4472136;   [0.5238096, -0.4732026, -0.7083099;
    #  0.0535898,  0.9732051, -0.2236068;    0.2827264,  0.8809524, -0.3794584;
    # -0.4472136,  0.2236068,  0.8660254]    0.8035480, -0.0014940,  0.5952381]
    print(rotm.size())
    print(rotm)
    print("============================================================")


def test_pose_to_transformation():
    print("============================================================")
    print("Test pose to 4x4 transformation matrix")
    print("============================================================")
    pose = np.array([1, 2, 3, 0.2341605, 0.468321, 0], dtype=np.float)
    tmat = utils.get_4x4_from_pose(pose, rotation_mode="axang")
    # the expected answer is
    # [0.8928203,  0.0535898,  0.4472136, 1;
    #  0.0535898,  0.9732051, -0.2236068, 2;
    # -0.4472136,  0.2236068,  0.8660254, 3;
    #  0.0,  0.0,  0.0, 1]

    print(tmat.shape)
    print(tmat)
    print("============================================================")


def test_pose_to_transformation_tensor():
    print("============================================================")
    print("Test pose to 4x4 transformation matrix for tensors")
    print("============================================================")

    pose_array = np.array([[1, 2, 3, 0.2341605, 0.468321, 0],
                           [9, 10, 11, 0.2285172, -0.914069, 0.4570345]], dtype=np.float)
    pose_tensor = torch.from_numpy(pose_array).to(DEVICE)
    tmat = utils.get_4x4_from_pose_tensor(pose_tensor, rotation_mode="axang")

    # the expected answer is
    # [0.8928203,  0.0535898,  0.4472136, 1;   [0.5238096, -0.4732026, -0.7083099, 9;
    #  0.0535898,  0.9732051, -0.2236068, 2;    0.2827264,  0.8809524, -0.3794584, 10;
    # -0.4472136,  0.2236068,  0.8660254, 3]    0.8035480, -0.0014940,  0.5952381, 11;
    #  0., 0, 0, 1.]                             0., 0., 0., 1.]

    print(tmat.size())
    print(tmat)
    print("============================================================")


def test_pose_error():
    print("============================================================")
    print("Test pose error")
    print("============================================================")

    pose_gt = np.array([[[1.0000000, 0.0000000, 0.0000000, 1],
                         [0.0000000, 0.7071068, -0.7071068, 2],
                         [0.0000000, 0.7071068, 0.707106, 3],
                         [0., 0., 0., 1.]],

                        [[1.0000000, 0.0000000, 0.0000000, 1],
                         [0.0000000, 0.7071068, -0.7071068, 2],
                         [0.0000000, 0.7071068, 0.707106, 3],
                         [0., 0., 0., 1.]]],dtype=np.float)
    pose_gt_tensor = torch.from_numpy(pose_gt).to(DEVICE)

    pose_est = np.array([[1.4, 2., 3, 0.6981317, 0, 0], [1., 2., 3.2, 0, 0, 2.3561945]], dtype=np.float)
    pose_est_tensor = torch.from_numpy(pose_est).to(DEVICE)

    from loss_functions import pose_loss

    mean_distance_error, mean_angle_error = pose_loss(pose_est_tensor, pose_gt_tensor)

    print(mean_distance_error)
    print(mean_angle_error)
    print("============================================================")

if __name__ == "__main__":
    test_axang_to_rotm()
    test_axang_to_rotm_tensor()

    test_euler_to_rotm()
    test_euler_to_rotm_tensor()

    test_quat_to_rotm()
    test_quat_to_rotm_tensor()

    test_pose_to_transformation()
    test_pose_to_transformation_tensor()

    test_pose_error()