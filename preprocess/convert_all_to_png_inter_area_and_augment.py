import epi_io
import os
import cv2
import numpy as np

import sys
sys.path.append('/media/dtejaswi/DATA/CommonData/Projects/student_projects/joe_daniel/code')
import utils


def get_index_for_saving(input_id, rotation=None, hflip=False):
    """
    This function correctly relocates the sub-apertures of the epi-module when a rotation or a flip is applied on it
    :param input_id: The id of the subaperture to be relocated
    :type input_id: int
    :param rotation: cv2.ROTATE_90_CLOCKWISE or cv2.ROTATE_90_COUNTERCLOCKWISE or cv2.ROTATE_180
    :type rotation: int or None
    :param hflip: Boolean indicating if a horizontal flip (left-right) has to be performed
    :type hflip: bool
    :return: the relocated index of the sub-aperture
    :rtype: int
    """
    if not hflip:
        # no horizontal flip
        if rotation is None:
            indices_for_saving = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        elif rotation == cv2.ROTATE_90_CLOCKWISE:
            indices_for_saving = [12, 11, 10, 9, 0, 1, 2, 3, 8, 13, 14, 15, 16, 7, 6, 4, 5]
        elif rotation is cv2.ROTATE_90_COUNTERCLOCKWISE:
            indices_for_saving = [4, 5, 6, 7, 16, 15, 14, 13, 8, 3, 2, 1, 0, 9, 10, 11, 12]
        elif rotation is cv2.ROTATE_180:
            indices_for_saving = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        else:
            raise ValueError("Unknown rotation type")
    else:
        if rotation is None:    # same as vflip -> 180
            indices_for_saving = [0, 1, 2, 3, 12, 11, 10, 9, 8, 7, 6, 5, 4, 13, 14, 15, 16]
        elif rotation == cv2.ROTATE_90_CLOCKWISE:   # same as vflip -> CCW 90
            indices_for_saving = [12, 11, 10, 9, 16, 15, 14, 13, 8, 3, 2, 1, 0, 7, 6, 5, 4]
        elif rotation is cv2.ROTATE_90_COUNTERCLOCKWISE:    # same as vflip -> CW 90
            indices_for_saving = [4, 5, 6, 7, 0, 1, 2, 3, 8, 13, 14, 15, 16, 9, 10, 11, 12]
        elif rotation is cv2.ROTATE_180:    # same as vflip
            indices_for_saving = [16, 15, 14, 13, 4, 5, 6, 7, 8, 9, 10, 11, 12, 3, 2, 1, 0]
        else:
            raise ValueError("Unknown rotation type")

    return indices_for_saving[input_id]


def get_seq_for_saving(input_seq, rotation=None, hflip=False):
    """
    Computes the sequence id for a given rotation and flip. The convention followed is
    [0-99] No flip, no rotation     [100-199] No flip, CW 90       [200-299] No flip, CCW 90      [300-399] No flip, 180
    [400-499] hflip, no rotation    [400-499] hflip, CW 90         [500-599] hflip, CCW 90        [600-699] hflip, 180

    :param input_seq: Input sequence ID
    :type input_seq: int
    :param rotation: cv2.ROTATE_90_CLOCKWISE or cv2.ROTATE_90_COUNTERCLOCKWISE or cv2.ROTATE_180
    :type rotation: int or None
    :param hflip: Boolean indicating if a horizontal flip (left-right) has to be performed
    :type hflip: bool
    :return: the new sequence id
    :rtype: int
    """
    if not hflip:
        # if no horizontal flip
        if rotation is None:
            return input_seq
        elif rotation == cv2.ROTATE_90_CLOCKWISE:
            return input_seq + 100
        elif rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
            return input_seq + 200
        elif rotation == cv2.ROTATE_180:
            return input_seq + 300
    else:
        if rotation is None:
            return input_seq + 400
        elif rotation == cv2.ROTATE_90_CLOCKWISE:
            return input_seq + 500
        elif rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
            return input_seq + 600
        elif rotation == cv2.ROTATE_180:
            return input_seq + 700


def apply_img_flip_and_rotation(img_orig, rotation=None, hflip=False):
    """
    Applies the specified flip followed by the specified rotation
    :param img_orig: Input image
    :type img_orig: ndarray
    :param rotation: cv2.ROTATE_90_CLOCKWISE or cv2.ROTATE_90_COUNTERCLOCKWISE or cv2.ROTATE_180
    :type rotation: int or None
    :param hflip: Boolean indicating if a horizontal flip (left-right) has to be performed
    :type hflip: bool
    :return: modified image
    :rtype: ndarray
    """
    if not hflip:
        # no horizontal flip
        img_flip = img_orig
    else:
        # apply horizontal flip
        img_flip = cv2.flip(img_orig, 1)

    if rotation is None:
        return img_flip
    elif rotation == cv2.ROTATE_90_CLOCKWISE:
        return cv2.rotate(img_flip, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return cv2.rotate(img_flip, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == cv2.ROTATE_180:
        return cv2.rotate(img_flip, cv2.ROTATE_180)


def apply_pose_rot_flip(input_rel_poses, rotation, hflip):
    """
    Applies the correct transformation to the relative poses give the specified rotation
    :param input_rel_poses: 4x4 transformation matrices of relative poses
    :type input_rel_poses: ndarray
    :param rotation: cv2.ROTATE_90_CLOCKWISE or cv2.ROTATE_90_COUNTERCLOCKWISE or cv2.ROTATE_180
    :type rotation: int or None
    :param hflip: boolean indicating if a horizontal flip has to be performed
    :type hflip: bool
    :return: The transformed relative poses
    :rtype: ndarray
    """
    if not hflip:
        if rotation is None:
            return input_rel_poses

    # transforms for rotation
    if rotation == cv2.ROTATE_90_CLOCKWISE:
        transform = np.array([[0., -1, 0., 0.],
                              [1., 0., 0., 0.],
                              [0., 0., 1., 0.],
                              [0., 0., 0., 1.]], dtype=np.float)
    elif rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
        transform = np.array([[0., 1, 0., 0.],
                              [-1., 0., 0., 0.],
                              [0., 0., 1., 0.],
                              [0., 0., 0., 1.]], dtype=np.float)
    elif rotation == cv2.ROTATE_180:
        transform = np.array([[-1., 0., 0., 0.],
                              [0., -1., 0., 0.],
                              [0., 0., 1., 0.],
                              [0., 0., 0., 1.]], dtype=np.float)
    elif rotation is None:
        transform = np.eye(4, dtype=np.float)
    else:
        raise ValueError("invalid rotation")

    transformed_rel_poses = input_rel_poses.copy()

    if hflip:
        # first apply flip and then the rotation
        flip = np.array([[-1., 0., 0., 0.],
                         [0., 1., 0., 0.],
                         [0., 0., 1., 0.],
                         [0., 0., 0., 1.]], dtype=np.float)
        transform = transform @ flip

    # apply the transform
    for idx in range(input_rel_poses.shape[0]):
        transformed_rel_poses[idx, :, :] = transform @ input_rel_poses[idx, :, :] @ transform.transpose()

    return transformed_rel_poses


def rotate_and_save_image(img_orig, img_id, camera_id, output_root_without_seq, seq,
                          rotation=None, hflip=False, is_seq_prefix=True):
    """
    Applies a flip followed by a rotation and saves the image

    :param img_orig: Downsampled image of size (256, 192)
    :type img_orig: opencv image
    :param img_id: index of the LF image in the sequence
    :type img_id: int
    :param camera_id: index of the sub aperture
    :type camera_id: int
    :param output_root_without_seq: output folder without the sequence ID
    :type output_root_without_seq: str
    :param seq: sequence ID
    :type seq: int
    :param rotation: rotation to apply on the image
    :type rotation: [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, None]
    :param hflip: apply horizontal flip or not
    :type hflip: bool
    :param is_seq_prefix: bool indicating if the input is a sequence or not
    :type is_seq_prefix: bool
    """

    img_mod = apply_img_flip_and_rotation(img_orig, rotation, hflip)
    id_for_saving = get_index_for_saving(camera_id, rotation, hflip)
    seq_rot_flip = get_seq_for_saving(seq, rotation, hflip)
    if is_seq_prefix:
        output_directory_per_camera = os.path.join(output_root_without_seq, "seq" + str(seq_rot_flip),
                                                   str(id_for_saving))
    else:
        output_directory_per_camera = os.path.join(output_root_without_seq, str(id_for_saving))

    if not os.path.exists(output_directory_per_camera):
        os.makedirs(output_directory_per_camera)
    output_filename = os.path.join(output_directory_per_camera, "{:010d}.png".format(img_id))
    cv2.imwrite(output_filename, img_mod)


def load_raw_poses(input_seq_dir, compute_for_rot_flip=True):
    """
    Process a directory of '000000x.txt, ....' absolute poses as recorded from the robot arm to a numpy arrays
    of various absolute and relative poses.
    :param input_seq_dir: Directory where the pose .txt files are present
    :type input_seq_dir: str
    :param compute_for_rot_flip: Boolean indicating if extra poses with flips and rotations have to be computed
    :type compute_for_rot_flip: bool
    :return:
    :rtype:
    """

    poses_base_endeff = np.empty((1, 4, 4))
    poses_first_endeff = np.empty((1, 4, 4))
    poses_prev_curr_endeff = np.empty((1, 4, 4))
    poses_base_cam = np.empty((1, 4, 4))
    poses_first_cam = np.empty((1, 4, 4))
    poses_prev_curr_cam = np.empty((1, 4, 4))

    # make a list of all the pose files in the folder and sort numerically
    pose_files = [f for f in os.listdir(input_seq_dir) if f.endswith(".txt")]
    pose_files = [os.path.join(input_seq_dir, f) for f in sorted(pose_files)]
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

        # ############################################################################################
        # 1, 2 - Pose of end effector wrt base and first pose
        # ############################################################################################
        if first_pose_of_endeff is None:
            first_pose_of_endeff = raw_pose  # this is done only in the very first run
        # pose of end effector wrt base is the raw pose
        p_base_endeff = utils.get_4x4_from_pose(raw_pose, rotation_mode='axang')
        # pose of end effector wrt first pose
        p_first_endeff = utils.get_relative_6dof(first_pose_of_endeff[0:3], first_pose_of_endeff[3:],
                                                 raw_pose[0:3], raw_pose[3:], rotation_mode='axang',
                                                 return_as_mat=True)

        # ############################################################################################
        # 3 - Current end effector pose relative to previous end effector pose
        # ############################################################################################
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

        # ############################################################################################
        # 4, 5 - Pose of camera wrt base and first pose
        # ############################################################################################
        # pose of camera wrt base
        p_base_cam = p_base_endeff @ t_endeff_cam
        if first_pose_of_cam is None:
            first_pose_of_cam = p_base_cam  # this is a 4x4 matrix
        # pose of camera wrt first pose
        p_first_cam = utils.get_relative_6dof(first_pose_of_cam[0:3, 3], first_pose_of_cam[0:3, 0:3],
                                              p_base_cam[0:3, 3], p_base_cam[0:3, 0:3], rotation_mode='rotm',
                                              return_as_mat=True)

        # ############################################################################################
        # 6 - Current camera pose relative to previous camera pose
        # ############################################################################################
        # current camera pose relative to previous camera pose
        if previous_pose_of_cam is None:
            p_prev_curr_cam = utils.get_relative_6dof(first_pose_of_cam[0:3, 3], first_pose_of_cam[0:3, 0:3],
                                                      p_base_cam[0:3, 3], p_base_cam[0:3, 0:3], rotation_mode='rotm',
                                                      return_as_mat=True)
        else:
            p_prev_curr_cam = utils.get_relative_6dof(previous_pose_of_cam[0:3, 3], previous_pose_of_cam[0:3, 0:3],
                                                      p_base_cam[0:3, 3], p_base_cam[0:3, 0:3], rotation_mode='rotm',
                                                      return_as_mat=True)
        previous_pose_of_cam = p_base_cam  # this is a 4x4 matrix

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

    # modify the relative poses applying flips and rotations if needed for data augmentation
    if compute_for_rot_flip:
        # no flip
        poses_prev_curr_cam_cw90 = apply_pose_rot_flip(poses_prev_curr_cam, cv2.ROTATE_90_CLOCKWISE, False)
        poses_prev_curr_cam_ccw90 = apply_pose_rot_flip(poses_prev_curr_cam, cv2.ROTATE_90_COUNTERCLOCKWISE, False)
        poses_prev_curr_cam_180 = apply_pose_rot_flip(poses_prev_curr_cam, cv2.ROTATE_180, False)

        # hflip
        poses_prev_curr_cam_hflip = apply_pose_rot_flip(poses_prev_curr_cam, None, True)
        poses_prev_curr_cam_hflip_cw90 = apply_pose_rot_flip(poses_prev_curr_cam, cv2.ROTATE_90_CLOCKWISE, True)
        poses_prev_curr_cam_hflip_ccw90 = apply_pose_rot_flip(poses_prev_curr_cam, cv2.ROTATE_90_COUNTERCLOCKWISE, True)
        poses_prev_curr_cam_hflip_180 = apply_pose_rot_flip(poses_prev_curr_cam, cv2.ROTATE_180, True)

        return (poses_base_endeff, poses_first_endeff, poses_prev_curr_endeff,
                poses_base_cam, poses_first_cam, poses_prev_curr_cam,
                poses_prev_curr_cam_cw90, poses_prev_curr_cam_ccw90, poses_prev_curr_cam_180, poses_prev_curr_cam_hflip,
                poses_prev_curr_cam_hflip_cw90, poses_prev_curr_cam_hflip_ccw90, poses_prev_curr_cam_hflip_180)

    return (poses_base_endeff, poses_first_endeff, poses_prev_curr_endeff,
            poses_base_cam, poses_first_cam, poses_prev_curr_cam)


def get_abs_from_rel_rot_flip(poses_prev_curr, pose_base_cam):
    poses_first_cam = utils.rel_to_abs(poses_prev_curr)
    poses_base_cam = np.zeros((poses_first_cam.shape[0], 4, 4), dtype=np.float)
    for idx in range(0, poses_first_cam.shape[0]):
        poses_base_cam[idx, :, :] = pose_base_cam @ poses_first_cam[idx, :, :]
    return poses_base_cam, poses_first_cam


def save_poses_rot_flip(seq, rotation, hflip, output_root_without_seq,
                        poses_base_endeff, poses_first_endeff, poses_prev_curr_endeff,
                        poses_prev_curr_cam_rot_flip, pose_base_cam_0):

    if rotation is None and hflip is False:
        raise ValueError("either one of rotation or flip has to be set")

    # get the sequence for saving
    seq_rot_flip = get_seq_for_saving(seq, rotation=rotation, hflip=hflip)
    # output folder
    output_dir = os.path.join(output_root_without_seq, "seq" + str(seq_rot_flip))
    # end effector poses
    np.save(os.path.join(output_dir, "poses_gt_base_endeff.npy"), poses_base_endeff)
    np.save(os.path.join(output_dir, "poses_gt_first_endeff.npy"), poses_first_endeff)
    np.save(os.path.join(output_dir, "poses_gt_prev_curr_endeff.npy"), poses_prev_curr_endeff)
    # compute camera poses
    poses_base_cam, poses_first_cam = get_abs_from_rel_rot_flip(poses_prev_curr_cam_rot_flip, pose_base_cam_0)
    np.save(os.path.join(output_dir, "poses_gt_base_cam.npy"), poses_base_cam)
    np.save(os.path.join(output_dir, "poses_gt_first_cam.npy"), poses_first_cam)
    np.save(os.path.join(output_dir, "poses_gt_prev_curr_cam.npy"), poses_prev_curr_cam_rot_flip)


if __name__ == "__main__":
    # ******************************************************
    # Script that converts epirect images to png images.
    # This code used CV.INTER_AREA interpolation
    # USAGE:
    # ******************************************************
    # The epirect images should be located in folders named seq#, where # is the number of the sequence.
    # Set the path to the input and output folders
    # Set the sequences that need to be converted

    # input_root = "/media/dtejaswi/Seagate Expansion Drive/JoeDanielThesisData/data/sequences"
    input_root = "/media/dtejaswi/DATA/CommonData/Projects/student_projects/joe_daniel/data/sequences/T/3"
    # input_root = "/media/dtejaswi/tejaswi1/joe_daniel_raw_recordings"
    output_root = "/media/dtejaswi/DATA/CommonData/Projects/student_projects/joe_daniel/data/module-1-1"

    # sequences = [3]
    sequences = [""]
    is_seq_prefix = False
    perform_augmentation = False
    only_poses = False
    only_images = True
    if perform_augmentation:
        save_dir = "module1-3-png"
    else:
        save_dir = "EPIdataset/extra4"

    # images
    for sequence in sequences:
        if is_seq_prefix:
            input_directory = os.path.join(input_root, "seq" + str(sequence))
            output_root_no_seq = os.path.join(output_root, save_dir)
        else:
            input_directory = os.path.join(input_root, str(sequence))
            output_root_no_seq = os.path.join(output_root, save_dir, str(sequence))

        print("epirect directory: {}".format(input_directory))
        print("output root: {}".format(output_root_no_seq))

        if not only_poses:

            # IMAGES
            # get a list of all the epirect images
            lfs = os.listdir(input_directory)
            lfs = [int(lf.split(".")[-2]) for lf in lfs if lf.split(".")[-1] == "epirect"]
            lfs = sorted(lfs)
            lfs = ["{:06d}.epirect".format(lf) for lf in lfs]
            lfs = [os.path.join(input_directory, lf) for lf in lfs]

            # go through the list of epirect images and convert them to png
            for i, lf in enumerate(lfs):
                print(f"\r{i}/{len(lfs)-1}", end='')

                # read rectified lightfied images as a list
                lf_img = epi_io.read_rectified(lf)

                # for each camera in the light field
                for j in range(0, len(lf_img)):
                    img = lf_img[j, :, :, :]
                    # downsample using area interpolation
                    img = cv2.resize(img, (256, 192), interpolation=cv2.INTER_AREA)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # normal image
                    rotate_and_save_image(img_orig=img, img_id=i, camera_id=j,
                                          output_root_without_seq=output_root_no_seq,
                                          seq=sequence, rotation=None, hflip=False, is_seq_prefix=is_seq_prefix)

                    if perform_augmentation:
                        # rotate clockwise by 90
                        rotate_and_save_image(img_orig=img, img_id=i, camera_id=j,
                                              output_root_without_seq=output_root_no_seq,
                                              seq=sequence, rotation=cv2.ROTATE_90_CLOCKWISE,
                                              hflip=False, is_seq_prefix=is_seq_prefix)
                        # rotate counterclockwise by 90
                        rotate_and_save_image(img_orig=img, img_id=i, camera_id=j,
                                              output_root_without_seq=output_root_no_seq,
                                              seq=sequence, rotation=cv2.ROTATE_90_COUNTERCLOCKWISE,
                                              hflip=False, is_seq_prefix=is_seq_prefix)
                        # rotate by 180
                        rotate_and_save_image(img_orig=img, img_id=i, camera_id=j,
                                              output_root_without_seq=output_root_no_seq,
                                              seq=sequence, rotation=cv2.ROTATE_180,
                                              hflip=False, is_seq_prefix=is_seq_prefix)

                        # horizontal filp
                        rotate_and_save_image(img_orig=img, img_id=i, camera_id=j,
                                              output_root_without_seq=output_root_no_seq,
                                              seq=sequence, rotation=None, hflip=True, is_seq_prefix=is_seq_prefix)
                        # horizontal filp, rotate clockwise by 90
                        rotate_and_save_image(img_orig=img, img_id=i, camera_id=j,
                                              output_root_without_seq=output_root_no_seq,
                                              seq=sequence, rotation=cv2.ROTATE_90_CLOCKWISE,
                                              hflip=True, is_seq_prefix=is_seq_prefix)
                        # horizontal filp, rotate counterclockwise by 90
                        rotate_and_save_image(img_orig=img, img_id=i, camera_id=j,
                                              output_root_without_seq=output_root_no_seq,
                                              seq=sequence, rotation=cv2.ROTATE_90_COUNTERCLOCKWISE,
                                              hflip=True, is_seq_prefix=is_seq_prefix)
                        # horizontal filp, rotate by 180
                        rotate_and_save_image(img_orig=img, img_id=i, camera_id=j,
                                              output_root_without_seq=output_root_no_seq,
                                              seq=sequence, rotation=cv2.ROTATE_180,
                                              hflip=True, is_seq_prefix=is_seq_prefix)

        # pose files
        if perform_augmentation:
            (ps_base_endeff, ps_first_endeff, ps_prev_curr_endeff, ps_base_cam, ps_first_cam, ps_prev_curr_cam,
             ps_prev_curr_cam_cw90, ps_prev_curr_cam_ccw90, ps_prev_curr_cam_180, ps_prev_curr_cam_flip,
             ps_prev_curr_cam_flip_cw90, ps_prev_curr_cam_flip_ccw90, ps_prev_curr_cam_flip_180) = \
                load_raw_poses(input_directory, compute_for_rot_flip=perform_augmentation)
        else:
            (ps_base_endeff, ps_first_endeff, ps_prev_curr_endeff,
             ps_base_cam, ps_first_cam, ps_prev_curr_cam) = load_raw_poses(input_directory,
                                                                           compute_for_rot_flip=perform_augmentation)

        # if os.path.exists(os.path.join(input_directory, "poses_gt_absolute.npy")) or os.path.exists(os.path.join(input_directory, "poses_gt_base_endeff.npy")):
        #     print("\nCopying GT absolute poses")

        if not only_images:
            # normal image
            seq_for_saving = get_seq_for_saving(sequence, rotation=None, hflip=False)
            if is_seq_prefix:
                output_directory = os.path.join(output_root_no_seq, "seq" + str(seq_for_saving))
            else:
                output_directory = os.path.join(output_root_no_seq)
            # shutil.copyfile(os.path.join(input_directory, "poses_gt_absolute.npy"),
            #                 os.path.join(output_directory, "poses_gt_absolute.npy"))
            np.save(os.path.join(output_directory, "poses_gt_base_endeff.npy"), ps_base_endeff)
            np.save(os.path.join(output_directory, "poses_gt_first_endeff.npy"), ps_first_endeff)
            np.save(os.path.join(output_directory, "poses_gt_prev_curr_endeff.npy"), ps_prev_curr_endeff)

            np.save(os.path.join(output_directory, "poses_gt_base_cam.npy"), ps_base_cam)
            np.save(os.path.join(output_directory, "poses_gt_first_cam.npy"), ps_first_cam)
            np.save(os.path.join(output_directory, "poses_gt_prev_curr_cam.npy"), ps_prev_curr_cam)

            if perform_augmentation:
                # rotate clockwise by 90
                save_poses_rot_flip(sequence, cv2.ROTATE_90_CLOCKWISE, False, output_root_no_seq,
                                    ps_base_endeff, ps_first_endeff, ps_prev_curr_endeff,
                                    ps_prev_curr_cam_cw90, ps_base_cam[0, :, :])
                # rotate counterclockwise by 90
                save_poses_rot_flip(sequence, cv2.ROTATE_90_COUNTERCLOCKWISE, False, output_root_no_seq,
                                    ps_base_endeff, ps_first_endeff, ps_prev_curr_endeff,
                                    ps_prev_curr_cam_ccw90, ps_base_cam[0, :, :])
                # rotate by 180
                save_poses_rot_flip(sequence, cv2.ROTATE_180, False, output_root_no_seq,
                                    ps_base_endeff, ps_first_endeff, ps_prev_curr_endeff,
                                    ps_prev_curr_cam_180, ps_base_cam[0, :, :])
                # hflip, no rotation
                save_poses_rot_flip(sequence, None, True, output_root_no_seq,
                                    ps_base_endeff, ps_first_endeff, ps_prev_curr_endeff,
                                    ps_prev_curr_cam_flip, ps_base_cam[0, :, :])
                # hflip, rotate clockwise by 90
                save_poses_rot_flip(sequence, cv2.ROTATE_90_CLOCKWISE, True, output_root_no_seq,
                                    ps_base_endeff, ps_first_endeff, ps_prev_curr_endeff,
                                    ps_prev_curr_cam_flip_cw90, ps_base_cam[0, :, :])
                # hflip, rotate counterclockwise by 90
                save_poses_rot_flip(sequence, cv2.ROTATE_90_COUNTERCLOCKWISE, True, output_root_no_seq,
                                    ps_base_endeff, ps_first_endeff, ps_prev_curr_endeff,
                                    ps_prev_curr_cam_flip_ccw90, ps_base_cam[0, :, :])
                # hflip, rotate by 180
                save_poses_rot_flip(sequence, cv2.ROTATE_180, True, output_root_no_seq,
                                    ps_base_endeff, ps_first_endeff, ps_prev_curr_endeff,
                                    ps_prev_curr_cam_flip_180, ps_base_cam[0, :, :])
            # else:
            #     print("GT absolute poses do not exist")
