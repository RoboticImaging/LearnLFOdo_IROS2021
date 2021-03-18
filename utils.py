from __future__ import division
import shutil
import numpy as np
import torch
# from path import Path
# import datetime
# from collections import OrderedDict
import os
# import time
import math
import pickle

from click.core import batch
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


def dump_config(save_path, args):
    """
    Writes current configuration to a pickle file called config.pkl and a text file called config.txt

    :param save_path: The path to the folder where the configuration should be written
    :type save_path: str
    :param args: The user input arguments
    :type args: argparse object
    """
    with open(os.path.join(save_path, "config.pkl"), 'wb') as pklfile:
        pickle.dump(args, pklfile)
    
    args = vars(args)
    dumpfile = open(os.path.join(save_path, "config.txt"), 'w')
    for key in args.keys():
        keystr = str(key).ljust(30, " ")
        dumpfile.write("{} {}\n".format(keystr, args[key]))


def load_config(cfgfile):
    """
    Loads a configuration file (.pkl)

    :param cfgfile: Config file (.pkl)
    :type cfgfile: str
    :return: the loaded configuration
    :rtype: Any
    """
    with open(cfgfile, 'rb') as pklfile:
        cfg = pickle.load(pklfile)

    return cfg


def make_save_path(args):
    """
    Creates a folder to save the results.

    :param args: The user input arguments
    :type args: argparse object
    :return: Path to the folder where the results will be saved
    :rtype: str
    """
    save_path = os.path.join(args.save_path, args.name)
    save_path = os.path.expanduser(save_path)
    os.makedirs(save_path, exist_ok=False)
    return save_path


def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    """
    Construct the list colormap, with interpolated values for higher resolution.
    For a linear segmented colormap, you can just specify the number of point in
    cm.get_cmap(name, lutsize) with the parameter lutsize

    :param low_res_cmap: The low resolution color map
    :type low_res_cmap: colormap object
    :param resolution: The resolution of the output high res color map
    :type resolution: int
    :param max_value: Maximum value in the output color map
    :type max_value: float
    :return: The high resolution colormap object
    :rtype: colormap object
    """
    x = np.linspace(0, 1, low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0, max_value, resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:, i]) for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


def opencv_rainbow(resolution=1000):
    """
    Constructs the opencv equivalent of Rainbow colormap

    :param resolution: Resolution of the output colormap
    :type resolution: int
    :return: Colormap
    :rtype: Colormap object
    """
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


# used in deprecated code, can be removed
def log_output_tensorboard(writer, prefix, index, suffix, n_iter, depth, disp, warped, diff, mask):
    disp_to_show = tensor2array(disp[0], max_value=None, colormap='magma')
    depth_to_show = tensor2array(depth[0], max_value=None)
    writer.add_image('{}/DisparityNormalized'.format(prefix), disp_to_show, n_iter)
    # writer.add_image('{} Depth Normalized {}/{}'.format(prefix, suffix, index), depth_to_show, n_iter)
    # log warped images along with explainability mask
    for j, (warped_j, diff_j) in enumerate(zip(warped, diff)):
        whole_suffix = '{} {}/{}'.format(suffix, j, index)
        warped_to_show = tensor2array(warped_j)
        diff_to_show = tensor2array(0.5*diff_j)
        writer.add_image('{}/Photometric warp'.format(prefix), warped_to_show, n_iter)
        writer.add_image('{}/Photometric error'.format(prefix), diff_to_show, n_iter)
        if mask is not None:
            mask_to_show = tensor2array(mask[0, j], max_value=1, colormap='bone')
            writer.add_image('{}/Exp mask Outputs'.format(prefix, whole_suffix), mask_to_show, n_iter)
        return


def tensor2array(tensor, max_value=None, colormap='rainbow'):
    """
    Converts tensor to array and applies a colormap to it? FOr visualization only

    :param tensor: the input tensor
    :type tensor: tensor
    :param max_value: the maximum value of the tensor
    :type max_value: float
    :param colormap: colormap to apply - rainbow, magma or bone
    :type colormap: str
    :return: the output array
    :rtype: array
    """

    colormaps = {'rainbow': opencv_rainbow(),
                 'magma': high_res_colormap(cm.get_cmap('magma')),
                 'bone': cm.get_cmap('bone', 10000)}

    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.dim() == 2 or tensor.size(0) == 1:
        norm_array = tensor.squeeze().numpy() / max_value
        array = colormaps[colormap](norm_array).astype(np.float32)
        array = array.transpose(2, 0, 1)
    elif tensor.dim() == 3:
        assert(tensor.size(0) == 3)
        array = 0.5 + tensor.numpy()*0.5
    else:
        raise ValueError("Dimensions of input tensor should be 1, 2 or 3")
    return array


# #############################
# CHECKPOINT UTILITIES
# #############################

def save_checkpoint(save_path, dispnet_state, exp_pose_state, is_best, filename='checkpoint.pth.tar'):
    """
    Saves a checkpoint

    :param save_path: Save path
    :type save_path: str
    :param dispnet_state: State of the dispnet network. Dictionary containing epoch and state_dict
    :type dispnet_state: dict
    :param exp_pose_state: State of the posenet network. Dictionary containing epoch and state_dict
    :type exp_pose_state: dict
    :param is_best: Boolean indicating if this is the best checkpoint in terms of training error
    :type is_best: bool
    :param filename: name of the checkpoint file
    :type filename: str
    """
    file_prefixes = ['dispnet', 'posenet']
    states = [dispnet_state, exp_pose_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, os.path.join(save_path, '{}_{}'.format(prefix, filename)))

    # If this is the best checkpoint, then also store an additional file with the suffix best
    if is_best:
        for prefix in file_prefixes:
            shutil.copyfile(os.path.join(save_path, '{}_{}'.format(prefix, filename)),
                            os.path.join(save_path, '{}_best.pth.tar'.format(prefix)))


def save_checkpoint_current(save_path, dispnet_state, exp_pose_state, epoch, filename='checkpoint.pth.tar'):
    """
    Saves a checkpoint at the specified epoch

    :param save_path: Save path
    :type save_path: str
    :param dispnet_state: State of the dispnet network. Dictionary containing epoch and state_dict
    :type dispnet_state: dict
    :param exp_pose_state: State of the posenet network. Dictionary containing epoch and state_dict
    :type exp_pose_state: dict
    :param epoch: current_epoch
    :type epoch: int
    :param filename: name of the checkpoint file
    :type filename: str
    """
    file_prefixes = ['dispnet', 'posenet']
    states = [dispnet_state, exp_pose_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, os.path.join(save_path, '{}_{}_{}'.format(prefix, epoch, filename)))


# #############################
# GEOMETRY AND LINEAR ALGEBRA
# #############################

def get_relative_6dof(p1, r1, p2, r2, rotation_mode='axang', correction=None, return_as_mat=False):
    """
    Returns relative pose between two frames, i.e. pose of frame 2 in frame 1.

    :param p1: position of frame 1 wrt the world frame
    :type p1: array (3 x 1) or (1 x 3)
    :param r1: orientation of frame 1 wrt the world frame
    :type r1: numpy.array
    :param p2: position of frame 2 wrt the world frame
    :type p2: array (3 x 1) or (1 x 3)
    :param r2: orientation of frame 2 wrt the world frame
    :type r2: numpy.array
    :param rotation_mode: Rotation convention for r1, r2. Can be 'axang', 'euler', 'rotm'. Default 'axang'
    :type rotation_mode: str
    :param correction: additional rotation matrix correction applied when camera was mounted on the arm differently
    :type correction: array
    :param return_as_mat: Boolean indicating if the relative pose has to be returned as a 4x4 transformation matrix
    :type correction: bool
    :return: Relative pose from frame 1 to frame 2. This is the pose of frame 2 in frame 1.
    :rtype: array (1 x 6) - translation and rotation as euler angles.
    """

    # Convert to rotation matrix
    if rotation_mode == "axang":
        r1 = axang_to_rotm(r1, with_magnitude=True)
        r2 = axang_to_rotm(r2, with_magnitude=True)
    elif rotation_mode == "euler":
        r1 = euler_to_rotm(r1[0], r1[1], r1[2])
        r2 = euler_to_rotm(r2[0], r2[1], r2[2])
    elif rotation_mode == "rotm":
        r1 = r1
        r2 = r2
    else:
        raise ValueError("Incorrect rotation_mode. Should be either axang, euler or rotm.")

    if correction is not None:
        r1 = r1 @ correction
        r2 = r2 @ correction

    # Ensure translations are column vectors
    p1 = np.float32(p1).reshape(3, 1)
    p2 = np.float32(p2).reshape(3, 1)

    # Concatenate to transformation matrices
    transformation1 = np.vstack([np.hstack([r1, p1]), [0, 0, 0, 1]])    # Brings a point in frame 1 to world frame
    transformation2 = np.vstack([np.hstack([r2, p2]), [0, 0, 0, 1]])    # Brings a point in frame 2 to world frame

    # The relative transformation is that one that brings a point in frame 2 to world frame and then to frame 1
    # i.e. transformation1.inv() x transformation2
    relative_pose = np.linalg.inv(transformation1) @ transformation2  # [4,4] transform matrix

    if return_as_mat:
        # return 4x4 transformation matrix
        return relative_pose
    else:
        # return x, y, x, euler angles
        return np.hstack((relative_pose[0:3, 3], rotm_to_euler(relative_pose[0:3, 0:3])))


def rotm_to_euler(rotm):
    """
    Converts rotation matrix to euler angles. DCM angles are decomposed into Z-Y-X euler rotations.

    :param rotm: Rotation matrix
    :type rotm: numpy.array
    :return: Euler angles corresponding to the rotation matrix
    :rtype: numpy.array
    """

    sy = math.sqrt(rotm[0, 0] * rotm[0, 0] + rotm[1, 0] * rotm[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rotm[2, 1], rotm[2, 2])
        y = math.atan2(-rotm[2, 0], sy)
        z = math.atan2(rotm[1, 0], rotm[0, 0])
    else:
        x = math.atan2(-rotm[1, 2], rotm[1, 1])
        y = math.atan2(-rotm[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def axang_to_rotm(r, with_magnitude=False):
    """
    Converts axis angle to rotation matrix.
    Expects 3-vector for with_magnitude=True.
    Expects 4-vector for with_magnitude=False.

    :param r: Rotation expressed in axis angle format
    :type r: 3-vector or 4-vector
    :param with_magnitude: Boolean indicating if the magnitude is included in the input vector or is separate
    :type with_magnitude: bool
    :return: Rotation matrix
    :rtype: ndarray
    """

    if with_magnitude:
        theta = np.linalg.norm(r) + 1e-15
        r = r / theta
        r = np.append(r, theta)

    kx, ky, kz, theta = r

    c = math.cos(theta)
    s = math.sin(theta)
    v = 1 - math.cos(theta)

    rotm = np.float32([
        [kx * kx * v + c, kx * ky * v - kz * s, kx * kz * v + ky * s],
        [kx * ky * v + kz * s, ky * ky * v + c, ky * kz * v - kx * s],
        [kx * kz * v - ky * s, ky * kz * v + kx * s, kz * kz * v + c]
    ])

    return rotm


def axang_to_rotm_tensor(vec, with_magnitude=True):
    """
    Converts an axis angle tensor to rotation matrix
    :param vec: tensor containing rotations expressed as axang [B, 3] or [B, 4]
    :type vec: tensor
    :param with_magnitude: Boolean indicating if the angle is the magnitude of the vector
    :type with_magnitude: bool
    :return: tensor containing rotation matrix [B, 3, 3]
    :rtype: tensor
    """
    if len(vec.size()) == 1:    # if input tensor is of size [3] or [4]
        vec = vec.unsqueeze(0)  # make it of size [1, 3] or [1, 4]

    batch_size = vec.size(0)

    if with_magnitude:
        theta = vec.norm(p=2, dim=1, keepdim=True) + 1e-15   # [B, 1]
        axis = vec / theta                                          # [B, 3]
        kx, ky, kz = axis[:, 0], axis[:, 1], axis[:, 2]             # [B]
        theta = theta.squeeze()                                     # [B]
    else:
        kx, ky, kz, theta = vec[:, 0], vec[:, 1], vec[:, 2], vec[:, 3]  # [B]

    c = torch.cos(theta)        # [B]
    s = torch.sin(theta)        # [B]
    v = 1 - torch.cos(theta)    # [B]

    rotm = torch.stack([kx * kx * v + c, kx * ky * v - kz * s, kx * kz * v + ky * s,
                        kx * ky * v + kz * s, ky * ky * v + c, ky * kz * v - kx * s,
                        kx * kz * v - ky * s, ky * kz * v + kx * s, kz * kz * v + c], dim=1).reshape(batch_size, 3, 3)
    return rotm


def euler_to_rotm(alpha, beta, gamma, rotation_order="zyx"):
    """
    Euler angle representation to rotation matrix. Rotation is composed in Z-Y-X order.

    :param alpha: rotation about x
    :type alpha: float
    :param beta: rotation about y
    :type beta: float
    :param gamma: rotation about z
    :type gamma: float
    :param rotation_order: order in which to compose the rotation - 'zyx' or 'xyz', default 'zyx'
    :type rotation_order: str
    :return: rotation matrix
    :rtype: numpy.array
    """

    rot_x = np.float32([
        [1, 0, 0],
        [0, math.cos(alpha), -math.sin(alpha)],
        [0, math.sin(alpha), math.cos(alpha)]
    ])

    rot_y = np.float32([
        [math.cos(beta), 0, math.sin(beta)],
        [0, 1, 0],
        [-math.sin(beta), 0, math.cos(beta)]
    ])

    rot_z = np.float32([
        [math.cos(gamma), -math.sin(gamma), 0],
        [math.sin(gamma), math.cos(gamma), 0],
        [0, 0, 1]
    ])

    if rotation_order == "zyx":
        rotm = rot_z @ rot_y @ rot_x
    elif rotation_order == "xyz":
        rotm = rot_x @ rot_y @ rot_z
    else:
        raise ValueError("only xyz and zyx rotation_order are implemented")
    return rotm


def euler_to_rotm_tensor(angle, rotation_order="zyx"):
    """
    Convert euler angles to rotation matrix.
    Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    :param angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    :type angle: tensor
    :param rotation_order: order in which to compose the rotation - 'zyx' or 'xyz', default 'zyx'
    :type rotation_order: str
    :return: Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    :rtype: tensor
    """
    if len(angle.size()) == 1:      # if input tensor is of size [3] or [4]
        angle = angle.unsqueeze(0)  # make it of size [1, 3] or [1, 4]

    batch_size = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]     # [B]

    zeros = torch.zeros(z.size(0), device=angle.device)  # [B]
    ones = torch.ones(z.size(0), device=angle.device)   # [B]

    rot_x = torch.stack([ones, zeros, zeros,
                        zeros,  torch.cos(x), -torch.sin(x),
                        zeros,  torch.sin(x),  torch.cos(x)], dim=1).reshape(batch_size, 3, 3)  # [B, 3, 3]

    rot_y = torch.stack([torch.cos(y), zeros,  torch.sin(y),
                         zeros,  ones, zeros,
                         -torch.sin(y), zeros,  torch.cos(y)], dim=1).reshape(batch_size, 3, 3)  # [B, 3, 3]

    rot_z = torch.stack([torch.cos(z), -torch.sin(z), zeros,
                         torch.sin(z), torch.cos(z), zeros,
                         zeros, zeros, ones], dim=1).reshape(batch_size, 3, 3)  # [B, 3, 3]

    if rotation_order == "zyx":
        rotm = rot_z @ rot_y @ rot_x        # [B, 3, 3]
    elif rotation_order == "xyz":
        rotm = rot_x @ rot_y @ rot_z        # [B, 3, 3]
    else:
        raise ValueError("only xyz and zyx rotation_order are implemented")
    return rotm     # [B, 3, 3]


def quat_to_rotm(quat, order="wxyz"):
    """
    Converts quaternion to a rotation matrix

    :param quat: quaternion in the order (w, x, y, z)
    :type quat: ndarray
    :param order: order of the elements of the quarternion "wxyz" or "xyzw", default "wxyz"
    :type order: str
    :return: Rotation matrix corresponding to the quaternion - 3x3
    :rtype: ndarray
    """

    # normalize the quaternion just to be sure
    norm_quat = quat / np.linalg.norm(quat)
    if order == "wxyz":
        w, x, y, z = norm_quat[0], norm_quat[1], norm_quat[2], norm_quat[3]
    elif order =="xyzw":
        x, y, z, w = norm_quat[0], norm_quat[1], norm_quat[2], norm_quat[3]
    else:
        raise ValueError("order should be wxyz or xyzw")

    w2, x2, y2, z2 = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rot_mat = np.array([[w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz],
                        [2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx],
                        [2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2]])
    return rot_mat


def quat_to_rotm_tensor(quat, order="wxyz"):
    """
    Converts quaternion to a rotation matrix

    :param quat: quaternion in the order (w, x, y, z) -- [B, 4]
    :type quat: tensor
    :param order: order of the elements of the quarternion "wxyz" or "xyzw", default "wxyz"
    :type order: str
    :return: Rotation matrix corresponding to the quaternion -- [B, 3, 3]
    :rtype: tensor
    """
    if len(quat.size()) == 1:       # if input tensor is of size [4]
        quat = quat.unsqueeze(0)    # make it of size [1, 4]

    batch_size = quat.size(0)

    # normalize the quaternion just to be sure
    norm_quat = quat / quat.norm(p=2, dim=1, keepdim=True)      # [B, 4]
    if order == "wxyz":
        w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]     # [B]
    elif order =="xyzw":
        x, y, z, w = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]  # [B]
    else:
        raise ValueError("order should be wxyz or xyzw")

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)     # [B]
    wx, wy, wz = w * x, w * y, w * z    # [B]
    xy, xz, yz = x * y, x * z, y * z    # [B]

    rot_mat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                           2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                           2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).reshape(batch_size, 3, 3)
    return rot_mat


def get_4x4_from_pose(pose, rotation_mode="axang"):
    """
    Converts a pose from [x,y,z,axis-angle with magnitude] or [x,y,z,euler angles] to a 4x4 transformation matrix
    :param pose: a 6 vector containing [x,y,z,axis angle with magnitude] or [x, y, z, euler angles]
    :type pose: array or list or ndarray
    :param rotation_mode: Rotation convention. Can be 'axang', 'euler'. Default 'axang'
    :type rotation_mode: str
    :return: 4x4 transformation matrix
    :rtype: np.array
    """
    t = np.array(pose[0:3]).reshape(1, 3)
    if rotation_mode == "axang":    # raw data from the UR5e are in this format
        rotm = axang_to_rotm([pose[3], pose[4], pose[5]], with_magnitude=True)
    elif rotation_mode == "euler":
        rotm = euler_to_rotm(pose[3], pose[4], pose[5])
    else:
        raise ValueError("Input mode should be axang or euler")
    tr = np.concatenate((rotm, np.transpose(t)), axis=1)
    tr = np.concatenate((tr, np.zeros((1, 4))), axis=0)
    tr[3, 3] = 1
    return tr


def get_4x4_from_pose_tensor(vec, rotation_mode='euler'):
    """
    Convert pose expressed as [B, 6] or [B, 7] to [B, 4, 4] transformation matrix.
    If [B, 7] then the quaternion is expected in the w, x, y, z order

    :param vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6] or [B, 7]
    :type vec: tensor
    :param rotation_mode: rotation notation ["axang","euler","quat"], default "axang"
    :type rotation_mode: str
    :return: A transformation matrix -- [B, 4, 4]
    :rtype: tensor
    """
    assert rotation_mode in ['axang', 'euler', 'quat']

    if len(vec.size()) == 2:    # [B, 6]
        translation = vec[:, :3].unsqueeze(2)  # [B, 3, 1]
        rot = vec[:, 3:]    # [B, 3]
        if rotation_mode == 'euler':
            rot_mat = euler_to_rotm_tensor(rot)  # [B, 3, 3]
        elif rotation_mode == 'quat':
            rot_mat = quat_to_rotm_tensor(rot, order="wxyz")  # [B, 3, 3]
        elif rotation_mode == 'axang':
            rot_mat = axang_to_rotm_tensor(rot, with_magnitude=True)  # [B, 3, 3]
        else:
            raise ValueError("rotation_mode should be 'axang', 'euler' or 'quat' ")

        transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
        last_row = torch.zeros(vec.size(0), 1, 4, device=vec.device)       # [B, 1, 4]
        last_row[:, :, 3] = 1.0
        transform_mat = torch.cat([transform_mat, last_row], dim=1)     # [B, 4, 4]

        return transform_mat
    else:   # [B, refs, 6]
        batch_size, num_refs, _ = vec.size()
        transform_mats = torch.zeros(batch_size, num_refs, 4, 4, device=vec.device)
        transform_mats[:, :, 3, 3] = 1.0
        for r in range(0, num_refs):
            translation = vec[:, r, :3]
            rot = vec[:, r, 3:]             # [B, 3]
            if rotation_mode == 'euler':
                rot_mat = euler_to_rotm_tensor(rot)  # [B, 3, 3]
            elif rotation_mode == 'quat':
                rot_mat = quat_to_rotm_tensor(rot, order="wxyz")  # [B, 3, 3]
            elif rotation_mode == 'axang':
                rot_mat = axang_to_rotm_tensor(rot, with_magnitude=True)  # [B, 3, 3]
            else:
                raise ValueError("rotation_mode should be 'axang', 'euler' or 'quat' ")

            transform_mats[:, r, 0:3, 0:3] = rot_mat
            transform_mats[:, r, 0:3, 3] = translation
        return transform_mats


def rel_to_abs(poses_prev_curr):
    """
    Converts an array of relative poses converts it to absolute poses wrt the first pose
    :param poses_prev_curr: nx4x4 numpy array with n 4x4 relative poses
    :type poses_prev_curr: np.array
    :return: nx4x4 numpy array of absolute poses wrt the first pose (origin)
    :rtype: np.array
    """
    poses_abs = np.zeros((poses_prev_curr.shape[0], 4, 4), dtype=np.float)
    poses_abs[0, :] = poses_prev_curr[0, :, :]

    tr_base_prev = poses_prev_curr[0, :, :]
    for i in range(1, poses_prev_curr.shape[0]):
        tr_base_prev = tr_base_prev @ poses_prev_curr[i, :, :]
        poses_abs[i, :, :] = tr_base_prev
    return poses_abs


# ########################
# PLOTTING UTILITIES
# ########################

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def draw_axes_at(pose, axes, scale=1.0, frame_label=0, arrowstyle='->'):
    # tr = get_4x4_from_pose(pose)
    tr = pose
    arrow_prop_dict = dict(mutation_scale=15, arrowstyle=arrowstyle, shrinkA=0, shrinkB=0)

    orig = [tr[0, 3], tr[1, 3], tr[2, 3]]
    xtip = [tr[0, 3] + scale * tr[0, 0],
            tr[1, 3] + scale * tr[1, 0],
            tr[2, 3] + scale * tr[2, 0]]
    ytip = [tr[0, 3] + scale * tr[0, 1],
            tr[1, 3] + scale * tr[1, 1],
            tr[2, 3] + scale * tr[2, 1]]
    ztip = [tr[0, 3] + scale * tr[0, 2],
            tr[1, 3] + scale * tr[1, 2],
            tr[2, 3] + scale * tr[2, 2]]

    a = Arrow3D([orig[0], xtip[0]], [orig[1], xtip[1]], [orig[2], xtip[2]], **arrow_prop_dict, color='r', linewidth=0.5)
    axes.add_artist(a)

    a = Arrow3D([orig[0], ytip[0]], [orig[1], ytip[1]], [orig[2], ytip[2]], **arrow_prop_dict, color='g', linewidth=0.5)
    axes.add_artist(a)

    a = Arrow3D([orig[0], ztip[0]], [orig[1], ztip[1]], [orig[2], ztip[2]], **arrow_prop_dict, color='b', linewidth=0.5)
    axes.add_artist(a)

    off = 0.1 * scale
    axes.text(orig[0], orig[1], orig[2] - off, r'${}$'.format(str(frame_label)), fontsize='x-small')
    axes.text(xtip[0] + off, xtip[1], xtip[2], r'$x$', fontsize='xx-small')
    axes.text(ytip[0], ytip[1] + off, ytip[2], r'$y$', fontsize='xx-small')
    axes.text(ztip[0], ztip[1], ztip[2] + off, r'$z$', fontsize='xx-small')
