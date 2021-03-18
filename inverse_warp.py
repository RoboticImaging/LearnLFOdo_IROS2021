from __future__ import division
import torch
from epimodule import get_transforms_subcam_centrecam_tensor
from epimodule import get_transforms_centrecam_subcam_tensor
import utils


pixel_coords = None


def set_id_grid(depth):
    """
    Generates a grid of successive integers (pixel coordinates) of the size of the input depth image and
    converts them to the same type as that of the depth image.
    NOTE: this function updates a global variable.

    :param depth: depth image
    :type depth: tensor
    :return: the grid of pixel coordinates
    :rtype: tensor
    """
    global pixel_coords
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1, h, w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def check_sizes(input_tensor, input_name, expected):
    """
    Check if the input if the the expected size.

    :param input_tensor: tensor whose size has to be checked
    :type input_tensor: tensor
    :param input_name: Name of the input tensor, just for printing
    :type input_name: str
    :param expected: Expected output specified as a string - Eg. B56 for Bx5x6, B1HW for Bx1xHxW
    :type expected: str
    """
    condition = [input_tensor.ndimension() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input_tensor.size(i) == int(size))
    assert(all(condition)),\
        "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected), list(input_tensor.size()))


def pixel2cam(depth, intrinsics_inv):
    """
    Back projects pixels in the depth image to points in the camera frame

    :param depth: depth maps -- [B, H, W]
    :type depth: tensor
    :param intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    :type intrinsics_inv: tensor
    :return: array of (u,v,1) cam coordinates -- [B, 3, H, W]
    :rtype: tensor
    """
    global pixel_coords

    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def cam2pixel(points_1, proj_rot_2_1, proj_tr_2_1):
    """
    Transform points in one camera frame to normalized pixels in another camera frame.

    :param points_1: coordinates of 3D points defined in the first camera coordinates system -- [B, 3, H, W]
    :type points_1:
    :param proj_rot_2_1: matrix to rotate points from frame 1 to frame 2 and projects them to frame 2 -- [B, 3, 3]
    :type proj_rot_2_1:
    :param proj_tr_2_1: vector to translate points from frame 1 to frame 2 and projects them to frame 2 -- [B, 3, 1]
    :type proj_tr_2_1:
    :return: array of [-1,1] normalized coordinates in image 2 -- [B, 2, H, W]
    :rtype: tensor
    """
    b, _, h, w = points_1.size()
    cam_coords_flat = points_1.reshape(b, 3, -1)  # [B, 3, H*W]
    # apply rotation
    if proj_rot_2_1 is not None:
        pcoords = proj_rot_2_1 @ cam_coords_flat
    else:
        pcoords = cam_coords_flat
    # apply translation
    if proj_tr_2_1 is not None:
        pcoords = pcoords + proj_tr_2_1  # [B, 3, H*W]

    x = pcoords[:, 0]
    y = pcoords[:, 1]
    z = pcoords[:, 2].clamp(min=1e-3)

    x_norm = 2*(x / z) / (w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    y_norm = 2*(y / z) / (h-1) - 1  # Idem [B, H*W]

    coords = torch.stack([x_norm, y_norm], dim=2)  # [B, H*W, 2]
    return coords.reshape(b, h, w, 2)


def inverse_warp(img, depth, pose, intrinsics, rotation_mode='axang', padding_mode='zeros'):
    """
    Given an image from a reference frame, the depth of the target frame, and the pose of the target frame relative
    to the reference frame, computes an "expected" reference image by warping the target image.
    This is done as follows:
    1. Back project pixels from the target image to the target camera frame using the depth image of the target frame
    2. This is the "world" from the target frame's point of view
    3. Use the pose of the target frame relative to the reference frame to transform the points into the reference frame
    4. Project these points back into the reference frame to get pixel location of the points.
    5. Sample the reference image at these locations to get the "expected" reference image.
    6. Additionally compute a mask of valid pixels, as some projected pixel locations may fall outside the image plane.

    :param img: the reference image (where to sample pixels) -- [B, 3, H, W]
    :type img: tensor
    :param depth: depth map of the target image -- [B, H, W]
    :type depth: tensor
    :param pose: relative pose of the target frame in the reference frame -- [B, 6]
    :type pose: tensor
    :param intrinsics: camera intrinsics matrix -- [B, 3, 3]
    :type intrinsics:  tensor
    :param rotation_mode: mode in which the rotation components are represented "axang", "euler" or "quat"
    :type rotation_mode: str
    :param padding_mode: padding mode for outside grid values - zeros, border or reflection. Default: zeros
    :type padding_mode: str
    :return: Expected (warped) reference image and Boolean array indicating valid pixels
    :rtype: tuple
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()
    # project pixels from target depth image to target camera frame
    points_tgtcam = pixel2cam(depth, intrinsics.inverse())  # [B,3,H,W]

    # get relative pose of the target frame in the reference frame. This transform when applied to the above points,
    # brings them to the reference frame
    posemat_srccam_tgtcam = utils.get_4x4_from_pose_tensor(pose, rotation_mode)  # [B,4,4]

    # Get projection matrix for tgt camera frame to source pixel frame (transformation + projection combined)
    proj_srcpixel_tgtcam = intrinsics @ posemat_srccam_tgtcam[:, 0:3, :]  # [B, 3, 4]
    rot, tr = proj_srcpixel_tgtcam[:, :, 0:3], proj_srcpixel_tgtcam[:, :, -1:]

    # project points from the target camera frame to the reference image plane.
    # The pixel values returned by cam2pixel are normalized to [-1,1].
    # (-1,-1) is top left corner and (1,1) is bottom right
    coords_srcpixel = cam2pixel(points_tgtcam, rot, tr)  # [B,H,W,2]

    # Sample the reference image at the above computed pixel coordinates to get the warped image
    warped_img = torch.nn.functional.grid_sample(img, coords_srcpixel, padding_mode=padding_mode, align_corners=False)
    # Compute mask of valid pixels
    valid_pixels = coords_srcpixel.abs().max(dim=-1)[0] <= 1

    return warped_img, valid_pixels


def inverse_multiwarp(ref_img, depth, pose_ref_tgt, intrinsics, padding_mode='zeros'):
    """
    Given an image from a reference frame (any sub-aperture), the depth of the target frame (same sub-aperture),
    and the pose of the target sub-aperture frame relative to the reference sub-aperture frame, computes an image that
    is the reference image warped to the target frame.

    This is done as follows:
    1. Back project pixels from the target image to the target camera frame using the depth image of the target frame
    2. This is the "world" from the target frame's point of view
    3. Use the pose of the target frame relative to the reference frame to transform the points into the reference frame
    4. Project these points back into the reference frame to get pixel location of the points.
    5. Sample the reference image at these locations to generate the reference image warped to the target view.
    6. Additionally compute a mask of valid pixels, as some projected pixel locations may fall outside the image plane.

    :param ref_img: the reference image (where to sample pixels) -- [B, 1, H, W]
    :type ref_img: tensor
    :param depth: depth map of the target image -- [B, H, W]
    :type depth:tensor
    :param pose: relative pose of the target frame in the reference frame as a 4x4 matrix -- [B, 4, 4]
    :type pose: tensor
    :param intrinsics: camera intrinsics matrix -- [B, 3, 3]
    :type intrinsics: tensor
    :param padding_mode: padding mode for outside grid values - zeros, border or reflection. Default: zeros
    :type padding_mode: str
    :return: Warped reference image [B, 1, H, W] and Boolean tensor indicating valid pixels [B, 1, H, W]
    :rtype: tuple
    """

    check_sizes(ref_img, 'img', 'B1HW')
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(pose_ref_tgt, 'pose', 'B44')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = ref_img.size()
    # project pixels from depth image to depth camera frame
    points_tgtcam = pixel2cam(depth, intrinsics.inverse())  # [B,3,H,W]
    # get relative pose of the target frame in the source frame. This transform when applied to the above points,
    # brings them to the source frame
    posemat_srccam_tgtcam = pose_ref_tgt[:, 0:3, :]         # [B, 3, 4]

    # Get projection matrix for tgt camera frame to reference pixel frame (transformation and projection)
    proj_srcpixel_tgtcam = intrinsics @ posemat_srccam_tgtcam  # [B, 3, 4]
    rot, tr = proj_srcpixel_tgtcam[:, :, 0:3], proj_srcpixel_tgtcam[:, :, -1:]

    # project points from the target camera frame to the reference image plane.
    # The pixel values returned by cam2pixel are normalized to [-1,1].
    # (-1,-1) is top left corner and (1,1) is bottom right
    coords_srcpixel = cam2pixel(points_tgtcam, rot, tr)  # [B,H,W,2]

    # Sample the reference image at the above computed pixel coordinates to get the warped image
    warped_img = torch.nn.functional.grid_sample(ref_img, coords_srcpixel,
                                                 padding_mode=padding_mode, align_corners=False)
    # Compute mask of valid pixels
    valid_points = coords_srcpixel.abs().max(dim=-1)[0] <= 1

    return warped_img, valid_points


def transform_refsub_tgtsub(pose_ref_tgt, cam_num):
    """
    Given the pose of the central sub-aperture of the tgt frame relative to the central sub-aperture of the ref frame
    and a chosen sub-aperture of the tgt frame, returns the pose of chosen sub-aperture of the tgt frame relative to the
    corresponding sub-aperture of the ref frame.

    If tgt frame is represented by 1 and ref frame by 2, give index of sub-aperture s1 and T_c2_c1,
    computes T_s2_s1 = T_s2_c2 x T_c2_c1 x T_c1_s1, where s2 is the corresponding sub-aperture of s1 in the ref frame.
    T_s2_c2 and T_c1_s1 are computed using the geometric structure of the lightfield camera array.

    :param pose_ref_tgt: pose of the central sub-aperture of the tgt frame relative to the central sub-aperture
     of the ref frame -- [B, 6], we shall consider this to be in axang format
    :type pose_ref_tgt: tensor
    :param cam_num: index of the chosen sub-aperture -- scalar integer -- [B]
    :type cam_num: tensor
    :return: pose of the chosen sub-aperture of the tgt frame relative to the corresponding sub-aperture of
    the ref frame
    :rtype: tensor
    """

    check_sizes(pose_ref_tgt, 'pose', 'B6')
    device = pose_ref_tgt.device

    # convert input pose to a 4x4 matrix
    posemat_ref_tgt = utils.get_4x4_from_pose_tensor(pose_ref_tgt, rotation_mode="axang")  # [B, 4, 4]

    # transform from chosen sub-aperture of target frame to central sub-aperture of target frame
    t_tgtcentral_tgtsub = get_transforms_centrecam_subcam_tensor(cam_num, device)
    # transform from central sub-aperture of reference frame to chosen sub-aperture of reference frame
    t_refsub_refcentral = get_transforms_subcam_centrecam_tensor(cam_num, device)

    return t_refsub_refcentral @ posemat_ref_tgt @ t_tgtcentral_tgtsub
