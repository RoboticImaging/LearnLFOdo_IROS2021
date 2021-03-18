import math
import numpy as np
import torch
import cv2
# from imageio import imread
import os
from utils import get_relative_6dof

CAMERA_SPACING = 0.04
DEFAULT_PATCH_SIZE = (160, 224)

# CAMERA_SPACING = 0.01
# DEFAULT_PATCH_SIZE = (160, 160)

DEFAULT_PATCH_INTRINSICS = np.array([
    [197.68828,     0,              DEFAULT_PATCH_SIZE[1]/2],
    [0,             197.68828,      DEFAULT_PATCH_SIZE[0]/2],
    [0,             0,              1]
]).astype(np.float32)


def load_as_float(path, gray, patch_size=None):
    """
    Loads image as a float array and takes a central crop.

    :param path: Path to the image to load
    :type path: str
    :param gray: Boolean indicating if the image has to be converted to grayscale
    :type gray: bool
    :param patch_size: size of the central crop, specified as integer, list or tuple
    :type patch_size: int, list, tuple
    :return: loaded image
    :rtype: numpy.array
    """
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(np.float32)
    # im = imread(path).astype(np.float32)

    # check the type of the patch size
    assert (type(patch_size) in [int, list, tuple]) or (patch_size is None)

    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    elif type(patch_size) in [list, tuple]:
        assert len(patch_size) == 2

    # take central crop
    if patch_size:
        h, w = im.shape[0:2]
        x_min = math.floor(w / 2 - patch_size[1] / 2)
        y_min = math.floor(h / 2 - patch_size[0] / 2)
        im = im[y_min:y_min+patch_size[0], x_min:x_min+patch_size[1], :]

    # convert to gray if necessary
    if gray:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    return im


def load_lightfield(path, cameras, gray, patch_size=DEFAULT_PATCH_SIZE):
    """
    Loads images that comprise the lightfield from the specified path, takes a central crop from each image and
    returns all of these images as a list

    :param path: path to the images to load
    :type path: str
    :param cameras: list of cameras that are part of the lightfield camera
    :type cameras: list if ints
    :param gray: boolean indicating if the image has to be converted to grayscale or not
    :type gray: bool
    :param patch_size: size of the central crop
    :type patch_size: int, list or tuple
    :return: list of images that constitute the lightfield image
    :rtype: list
    """
    imgs = []
    for cam in cameras:
        img_path = path.replace('/8/', '/{}/'.format(cam))  # Hacky way to load the images. TODO: clean this up
        imgs.append(load_as_float(img_path, gray, patch_size=patch_size))

    return imgs


def load_relative_pose(frame_1, frame_2, no_pose=False):
    """
    Loads the pose of frame_2 wrt frame_1, i.e pose_1_2

    :param frame_1: path to the image frame 1
    :type frame_1: str
    :param frame_2: path to the image frame 2
    :type ref: str
    :param no_pose: Boolean, if True returns a zero relative pose.
    :type no_pose: bool
    :return: Relative pose of frame 2 wrt frame 1 as a 4x4 transformation matrix
    :rtype: ndarray
    """
    if no_pose:
        return np.eye(4, dtype=np.float)
    else:
        # Get the number in the filename - super hacky
        sequence_name = os.path.join("/", *frame_2.split("/")[:-2])
        pose_w_2_id = int(frame_2.split("/")[-1].split(".")[-2])
        pose_w_1_id = int(frame_1.split("/")[-1].split(".")[-2])
        # pose_file = np.load(os.path.join(sequence_name, "poses_gt_absolute.npy"))
        pose_file = np.load(os.path.join(sequence_name, "poses_gt_base_cam.npy"))
        pose_2 = pose_file[pose_w_2_id, :]
        pose_1 = pose_file[pose_w_1_id, :]
        pose_1_2 = get_relative_6dof(pose_1[0:3, 3], pose_1[0:3, 0:3],
                                     pose_2[0:3, 3], pose_2[0:3, 0:3],
                                     rotation_mode='rotm', return_as_mat=True)
        return pose_1_2


def get_transform_subcam_centre(cam_index, camera_spacing):
    """
    Returns the transformation matrix that bring points represented in the central sub-aperture frame to the
    chosen sub-aperture frame. This is nothing but the pose of the central sub-aperture wrt. the chosen sub-aperture.

    :param cam_index: index of the sub-aperture
    :type cam_index: int
    :param camera_spacing: spacing between the cameras
    :type camera_spacing: float
    :return: 4x4 transformation matrix
    :rtype: numpy.array
    """
    if cam_index in [0, 1, 2, 3]:
        y = camera_spacing * (4 - cam_index)
        t_mat = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, y],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    elif cam_index in [13, 14, 15, 16]:
        y = camera_spacing * (12 - cam_index)
        t_mat = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, y],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    elif cam_index in [4, 5, 6, 7, 8, 9, 10, 11, 12]:
        x = camera_spacing * (8 - cam_index)
        t_mat = np.array([
            [1, 0, 0, x],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    else:
        raise ValueError("Camera must be in range [0, 16]")
    return t_mat


def get_transforms_subcam_centrecam(cameras, camera_spacing=CAMERA_SPACING):
    """
    Returns the transformation matrix that bring points represented in the central sub-aperture frame to the
    chosen sub-aperture frames. This is nothing but the pose of the central sub-aperture wrt. the chosen sub-apertures.

    :param cameras: tensor of which camera(s) to transform. Cameras are indexed as shown -- [B]
    :type cameras: tensor
    :param camera_spacing: spacing in between sub-apertures. Assumed isotropic in s and t.
    this shows the view from BEHIND the camera, i.e. optical axis going into the page.
                        0
                        1
                        2
                        3
            4  5  6  7  8  9  10 11 12
                        13               z
                        14              /
                        15             /_____ x
                        16             |
                                       |
                                       y
    :type camera_spacing: float
    :return: [4,4] homogeneous transformation matrix that bring points represented in the central sub-aperture frame
     to the chosen sub-aperture frames. This is also the pose of the central sub-aperture wrt. the chosen sub-apertures.
    :rtype: tensor
    """
    t_mats = []
    for cam in cameras:
        t_mats.append(get_transform_subcam_centre(cam, camera_spacing))

    return torch.Tensor(t_mats)


def get_transforms_centrecam_subcam(cameras, camera_spacing=CAMERA_SPACING):
    """
    Returns the transformation matrix that bring points represented in the chosen sub-aperture frames
    to the central sub-aperture frame. This is also the pose of the chosen sub-apertures wrt. the central sub-aperture.

    :param cameras: tensor of which camera(s) to transform. Cameras are indexed as shown -- [B]
    :type cameras: tensor
    :param camera_spacing: spacing in between sub-apertures. Assumed isotropic in s and t.
    this shows the view from BEHIND the camera, i.e. optical axis going into the page.
                        0
                        1
                        2
                        3
            4  5  6  7  8  9  10 11 12
                        13               z
                        14              /
                        15             /_____ x
                        16             |
                                       |
                                       y
    :type camera_spacing: float
    :return: [4,4] homogeneous transformation matrix that bring points represented in the chosen sub-aperture frames
    to the central sub-aperture frame. This is also the poses of the chosen sub-apertures wrt. the central sub-aperture.
    :rtype: tensor
    """
    t_mats = []
    for cam in cameras:
        t_mats.append(get_transform_subcam_centre(cam, -1.0 * camera_spacing))

    return torch.Tensor(t_mats)


def get_transforms_subcam_centrecam_tensor(cameras, device, camera_spacing=CAMERA_SPACING):
    """
    Returns the transformation matrix that bring points represented in the central sub-aperture frame to the
    chosen sub-aperture frames. This is nothing but the pose of the central sub-aperture wrt. the chosen sub-apertures.

    :param cameras: tensor of which camera(s) to transform. Cameras are indexed as shown -- [B]
    :type cameras: tensor
    :param device: The torch device on which to create the tensors
    :type device: pytorch device
    :param camera_spacing: spacing in between sub-apertures. Assumed isotropic in s and t.
    :type camera_spacing: float
    :return: [4,4] homogeneous transformation matrix that bring points represented in the central sub-aperture frame
     to the chosen sub-aperture frames. This is also the pose of the central sub-aperture wrt. the chosen sub-apertures.
    :rtype: tensor
    """

    # create a tensor with identity matrices of shape [B, 4, 4]
    tmats = torch.eye(4, device=device).unsqueeze(0).repeat(cameras.size(0), 1, 1)

    for b, cam_index in enumerate(cameras):
        if cam_index in [0, 1, 2, 3]:  # top 4 vertical sub-apertures. add positive y shift
            tmats[b, 1, 3] = camera_spacing * (4 - cam_index)
        elif cam_index in [13, 14, 15, 16]:  # bottom 4 vertical sub-apertures. add negative y shift
            tmats[b, 1, 3] = camera_spacing * (12 - cam_index)
        elif cam_index in [4, 5, 6, 7, 8, 9, 10, 11, 12]:  # horizontal sub-apertures. add appropriate x shift
            tmats[b, 0, 3] = camera_spacing * (8 - cam_index)
        else:
            raise ValueError("Camera must be in range [0, 16]")
    return tmats


def get_transforms_centrecam_subcam_tensor(cameras, device, camera_spacing=CAMERA_SPACING):
    """
    Returns the transformation matrix that bring points represented in the chosen sub-aperture frames
    to the central sub-aperture frame. This is also the pose of the chosen sub-apertures wrt. the central sub-aperture.

    :param cameras: tensor of which camera(s) to transform. Cameras are indexed as shown -- [B]
    :type cameras: tensor
    :param device: The torch device on which to create the tensors
    :type device: pytorch device
    :param camera_spacing: spacing in between sub-apertures. Assumed isotropic in s and t.
    :type camera_spacing: float
    :return: [4,4] homogeneous transformation matrix that bring points represented in the chosen sub-aperture frames
    to the central sub-aperture frame. This is also the poses of the chosen sub-apertures wrt. the central sub-aperture.
    :rtype: tensor
    """

    # create a tensor with identity matrices of shape [B, 4, 4]
    tmats = torch.eye(4, device=device).unsqueeze(0).repeat(cameras.size(0), 1, 1)

    for b, cam_index in enumerate(cameras):
        if cam_index in [0, 1, 2, 3]:           # top 4 vertical sub-apertures. add negative y shift
            tmats[b, 1, 3] = camera_spacing * (cam_index - 4)
        elif cam_index in [13, 14, 15, 16]:     # bottom 4 vertical sub-apertures. add positive y shift
            tmats[b, 1, 3] = camera_spacing * (cam_index - 12)
        elif cam_index in [4, 5, 6, 7, 8, 9, 10, 11, 12]:   # horizontal sub-apertures. add appropriate x shift
            tmats[b, 0, 3] = camera_spacing * (cam_index - 8)
        else:
            raise ValueError("Camera must be in range [0, 16]")
    return tmats

def shift_sum(lf, shift, dof, gray):
    """
    Computes the focal stack image given a set of images that form the lightfield.
    The focal stack is computed by computing the average of the images of the lightfield after
    shifting each image as desired. The shift is related to the synthetic focus of the camera.

    :param lf: list of images that constitutes the lightfield image
    :type lf: list
    :param shift: The amount of shift. This relates to where the camera should focus (synthetically)
    :type shift: int
    :param dof: depth of field. 17 = all cameras, 13,9 or 5 = leave out 1,2 or 3 cameras at the ends respectively
    :type dof: int
    :param gray: boolean indicating if the returned focal stack image should be converted to grayscale
    :type gray: bool
    :return: lightfield as a focal stack image
    :rtype: numpy array
    """
    lf = np.array(lf)
    assert (lf.shape[0] == 17)
    # lf[:,:,:,[0,2]] = lf[:,:,:,[2,0]]   # RGB to BGR - only for visualization does not really matter for code

    if dof == 17:
        left = [4, 5, 6, 7]
        right = [9, 10, 11, 12]
        top = [0, 1, 2, 3]
        bottom = [13, 14, 15, 16]
    elif dof == 13:
        left = [5, 6, 7]
        right = [9, 10, 11]
        top = [1, 2, 3]
        bottom = [13, 14, 15]
    elif dof == 9:
        left = [6, 7]
        right = [9, 10]
        top = [2, 3]
        bottom = [13, 14]
    elif dof == 5:
        left = [7]
        right = [9]
        top = [3]
        bottom = [13]
    else:
        raise ValueError("Cannot focus at depth {}".format(dof))

    focalstack = lf[8].astype(np.float32)
    if shift == 0:
        focalstack += np.sum(lf[left + right + top + bottom], 0)
    else:
        for i in left:
            # shift img to the left and add it to the focal stack
            img = lf[i]
            s = (8 - i) * shift
            focalstack[:, :-s:, :] += img[:, s:, :]
        for i in right:
            # shift img to the right and add it to the focal stack
            img = lf[i]
            s = (i - 8) * shift
            focalstack[:, s:, :] += img[:, :-s, :]
        for i in top:
            # shift img to the top and add it to the focal stack
            img = lf[i]
            s = (4 - i) * shift
            focalstack[:-s, :, :] += img[s:, :, :]
        for i in bottom:
            # shift img to the bottom and add it to the focal stack
            img = lf[i]
            s = (i - 12) * shift
            focalstack[s:, :, :] += img[:-s, :, :]

    focalstack = focalstack / dof   # compute average
    if gray:
        focalstack = cv2.cvtColor(focalstack, cv2.COLOR_RGB2GRAY)
    return focalstack.astype(np.uint8)


def load_multiplane_focalstack(path, num_planes, num_cameras, gray):
    """
    Loads a multiplane focal stack

    :param path: path to the image
    :type path: str
    :param num_planes: number of planes the focal stack should focus on 3, 5, 7, 9
    :type num_planes: int
    :param num_cameras: number of cameras to use in the lightfield image 17, 13, 9, 5
    :type num_cameras: int
    :param gray: boolean indicating if the image should be converted to grayscale or not
    :type gray: bool
    :return: list of images that constitute the multiplane focal stack
    :rtype: list
    """
    assert num_cameras in [5, 9, 13, 17]
    assert num_planes in [9, 7, 5, 3]

    planes = None   # This is also the amount of shift to use to compute the focal stack image
    if num_planes == 9:
        planes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    elif num_planes == 7:
        planes = [0, 1, 2, 3, 4, 6, 8]
    elif num_planes == 5:
        planes = [0, 2, 4, 6, 8]
    elif num_planes == 3:
        planes = [0, 4, 8]

    stacks = []
    lf = load_lightfield(path, gray=False, cameras=list(range(0, 17)))
    for p in planes:
        stacks.append(shift_sum(lf=lf, shift=p, dof=num_cameras, gray=gray))

    return stacks


def load_lightfield_horizontal_cameras(path, patch_size=None, rotate_ccw_by_90 = False):
    """
    Loads images only from the horizontal cameras

    :param path: path to the image
    :type path: str
    :param patch_size: size of the central crop
    :type patch_size: int, list, tuple
    :return: list of images that constitute the Lightfield image from only the horizontal cameras
    :rtype: list
    """
    # TODO: Why is camera 4 missing? Something to do with number of cameras.
    horizontal_cameras = [5, 6, 7, 8, 9, 10, 11, 12]
    lf = load_lightfield(path, gray=True, cameras=horizontal_cameras, patch_size=patch_size)
    lf = np.array(lf)
    h, w = lf.shape[1:3]

    if rotate_ccw_by_90:
        lf = lf.transpose([0, 2, 1])
    return lf


def load_lightfield_vertical_cameras(path, patch_size=None):
    """
    Loads images only from the vertical cameras

    :param path: path to the image
    :type path: str
    :param patch_size: size of the central crop
    :type patch_size: int, list, tuple
    :return: list of images that constitute the Lightfield image from only the vertical cameras
    :rtype: list
    """
    # TODO: Why is camera 0 missing? Something to do with number of cameras.
    vertical_cameras = [1, 2, 3, 8, 13, 14, 15, 16]
    lf = load_lightfield(path, gray=True, cameras=vertical_cameras, patch_size=patch_size)
    lf = np.array(lf)
    h, w = lf.shape[1:3]

    return lf


def load_tiled_epi_vertical(path, patch_size=DEFAULT_PATCH_SIZE):
    """
    Loads a tiled epipolar plane image (2D) from vertically displaced cameras.
    Corresponding columns of each camera are tiled horizontally to form epipolar plane "strips".
    Strips for each column are then tiled horizontally to form the resultant "wide" tiled EPI.

    :param path: path to the image of the central camera of the lightfield
    :type path: str
    :param patch_size: size of the central crop
    :type patch_size: int, list, tuple
    :return: list of vertical epipolar images in a tiled form   [B x nrows x (ncams * ncols)]
    :rtype: list
    """
    vertical = load_lightfield_vertical_cameras(path, patch_size).transpose(2, 0, 1)
    vertical = vertical.reshape(vertical.shape[1]*vertical.shape[0], vertical.shape[2], 1).transpose()
    return vertical


def load_tiled_epi_horizontal(path, patch_size=DEFAULT_PATCH_SIZE):
    """
    Loads a tiled epipolar plane image (2D) from horizontally displaced cameras.
    Corresponding rows of each camera are tiled vertically to form epipolar plane "strips".
    Strips for each row are then tiled vertically to form the resultant "tall" tiled EPI.

    :param path: path to the image of the central camera of the lightfield
    :type path: str
    :param patch_size: size of the central crop
    :type patch_size: int, list, tuple
    :return: list of horizontal epipolar images in a tiled form     [B x (nrows * ncams) x ncols]
    :rtype: list
    """
    horizontal = load_lightfield_horizontal_cameras(path, patch_size).transpose(1, 0, 2)
    horizontal = horizontal.reshape(1, horizontal.shape[0] * horizontal.shape[1], horizontal.shape[2])
    return horizontal


def load_tiled_epi_full(path, patch_size=DEFAULT_PATCH_SIZE):
    """
    Loads the tiled epipolar plane image (2D) from both vertically and horizontally displaced cameras.
    NOTE: The central camera is repeated twice.

    :param path: path to the image of the central camera of the lightfield
    :type path: str
    :param patch_size: size of the central crop
    :type patch_size: int, list, tuple
    :return: list of vertical epipolar images in a tiled form and list of horizontal epipolar images in tiled form
    :rtype: list, list
    """
    vertical = load_tiled_epi_vertical(path, patch_size)
    horizontal = load_tiled_epi_horizontal(path, patch_size)

    return vertical, horizontal


def load_stacked_epi(path, patch_size=DEFAULT_PATCH_SIZE, same_parallax=False):
    """
    Loads an epipolar volume. Using same_parallax works only works if the patch is a square.
    NOTE: When this function is used, the central camera is repeated twice. Use load_stacked_epi_no_repeats, if the
    central camera must not be repeated.

    :param path: path to the image
    :type path: str
    :param patch_size: size of the central crop - should be a square
    :type patch_size: int, list, tuple
    :param same_parallax: If True then the horizontal images are rotated CCW by 90 degrees
    so that the direction of parallax is the same as that of the vertical images
    :type same_parallax: bool
    :return: concatenation of horizontal and vertical epipolar plane images
    :rtype: numpy.array
    """
    assert isinstance(patch_size, int)
    horizontal = load_lightfield_horizontal_cameras(path, patch_size=patch_size, rotate_ccw_by_90=same_parallax)
    vertical = load_lightfield_vertical_cameras(path, patch_size=patch_size)
    epi = np.concatenate([horizontal, vertical], 0)
    return epi


def load_stacked_epi_no_repeats(path, patch_size=DEFAULT_PATCH_SIZE, same_parallax=False):
    """
    # Loads an epipolar volume. Using same_parallax works only works if the patch is a square.
    # NOTE: When using this function, the central camera is not repeated twice. Use load_stacked_epi, if the central
    camera must be repeated.

    :param path: path to the image
    :type path: str
    :param patch_size: size of the central crop - should be a square when same_parallax is set to true.
    :type patch_size: int, list, tuple
    :param same_parallax: If True then the horizontal images are rotated CCW by 90 degrees
    so that the direction of parallax is the same as that of the vertical images
    :type same_parallax: bool
    :return: concatenation of horizontal and vertical epipolar plane images
    :rtype: numpy.array
    """
    if same_parallax:
        assert isinstance(patch_size, int)

    vertical_cameras = [0, 1, 2, 3, 8, 13, 14, 15, 16]
    lfv = load_lightfield(path, gray=True, cameras=vertical_cameras, patch_size=patch_size)
    lfv = np.array(lfv)

    horizontal_cameras = [4, 5, 6, 7, 9, 10, 11, 12]  # we omit the central camera - 8
    lfh = load_lightfield(path, gray=True, cameras=horizontal_cameras, patch_size=patch_size)
    lfh = np.array(lfh)
    if same_parallax:   # rotate the horizontal images CCW by 90 degrees
        lfh = lfh.transpose([0, 2, 1])

    epi = np.concatenate([lfh, lfv], 0)
    return epi


# Demo of how to use shiftsum
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    lf1 = load_multiplane_focalstack(
        "/home/joseph/Documents/thesis/epidata/module-1-1/module1-1-png/seq2/8/0000000030.png",
        num_planes=9, num_cameras=9, gray=False)

    plt.imshow(lf1[8])
    plt.show()
