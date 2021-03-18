from __future__ import division
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from inverse_warp import inverse_warp, inverse_multiwarp
from inverse_warp import transform_refsub_tgtsub
import utils
import numpy as np

def multiwarp_photometric_loss(
    tgt_lf, ref_lfs, intrinsics, depth, poses_est_refs_tgt, metadata,
    rotation_mode='axang', padding_mode='zeros', sum_or_mean="sum"):
    """
    Computes photometric reconstruction loss across an entire lightfield.

    :param tgt_lf: Target lightfield images - this is a tensor of images [B, Ncams, H, W]
    :type tgt_lf: tensor
    :param ref_lfs: Reference lightfield images - this is a list of tensors of reference images [[B, Ncams, H, W]]
    :type ref_lfs: list
    :param intrinsics: Intrinsics matrix [B, 3, 3]
    :type intrinsics: array
    :param depth: Depth map of the target lightfield images at multiple scales of the scale pyramid - inverse of
    output of dispnet [B, NCams, H, W]
    :type depth: tuple or list
    :param poses_est_refs_tgt: Poses of the central aperture of the target lightfield relative to that of the reference
    lightfields. [B, SeqLen-1, 6]. This is the output of the pose network and we consider it to be in axang notation
    :type poses_est_refs_tgt: tensor
    :param metadata: Additional metadata
    :type metadata: dict
    :param rotation_mode: format in which the rotation components of the pose are specified. "euler" or "quat"
    :type rotation_mode: str
    :param padding_mode: padding mode for outside grid values - zeros, border or reflection. Default: zeros
    :type padding_mode: str
    :return: the photometric reconstruction loss, warped reference images, difference between tgt and warped ref images
    :rtype: float, list, list
    """

    def one_scale(depth_tgt):
        """
        Computes the reconstruction loss between the target_image and all the reference images. This is the sum of the
        mean of the absolute photometric error between the target image and the warped reference images, given a depth
        predition of the the target image. This target depth frame here is at one scale of the scale pyramid. Hence the
        name one_scale.

        :param depth_tgt: depth of the target sub-apertures of shape [batch_size x num_cams x hxw]
        :type depth_tgt: tensor or array
        :return: the reconstruction loss for
        :rtype: float
        """
        # check that there are as many poses as reference images per batch
        assert(poses_est_refs_tgt.size(1) == len(ref_lfs))

        reconstruction_loss = 0
        b, n, h, w = depth_tgt.size()
        downscale = tgt_lf.size(2)/h

        # scale the target and reference images to match the size of the depth image by performing interpolation
        tgt_lf_scaled = F.interpolate(tgt_lf, (h, w), mode='area')
        ref_lf_scaled = [F.interpolate(ref_lf, (h, w), mode='area') for ref_lf in ref_lfs]
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)

        warped_images = []
        diff_images = []

        # For every camera (sub aperture of the lightfield), perform the inverse warp
        for idx, cam in enumerate(metadata['cameras']):     # cam is of shape [B]
            # depth image of the chosen target sub-aperture
            depth_tgtsub = depth_tgt[:, idx, :, :]

            # For every reference image in the sequence 
            for i, ref_img in enumerate(ref_lf_scaled):
                # pose of the target sub-aperture frame relative to the corresponding reference sub-aperture frame
                pose_refsub_tgtsub = transform_refsub_tgtsub(poses_est_refs_tgt[:, i, :], cam)
                # reference image of the corresponding sub-aperture
                refsub_img = ref_img[:, idx:idx+1, :, :]
                # warp the reference image of the corresponding sub-aperture to the chosen target sub-aperture frame
                # and also get the mask of valid points
                ref_image_warped, valid_points = inverse_multiwarp(refsub_img, depth_tgtsub, pose_refsub_tgtsub,
                                                                   intrinsics_scaled, padding_mode)
                # compute the difference between the target image and the warped reference image, with mask applied
                diff = (tgt_lf_scaled[:, idx:idx+1, :, :] - ref_image_warped) * valid_points.unsqueeze(1).float()
                # compute the mean of the absolute difference and add it to the reconstruction loss
                reconstruction_loss += diff.abs().mean()        # this is already mean over batch size
                warped_images.append(ref_image_warped)
                diff_images.append(diff)

            if sum_or_mean == "mean":
                reconstruction_loss /= len(ref_lfs)     # divide by the number of reference images in the sequence

        if sum_or_mean == "sum":
            return reconstruction_loss, warped_images, diff_images
        elif sum_or_mean == "mean":     # divide by the number of sub-apertures considered for the warp
            return reconstruction_loss / len(metadata['cameras']), warped_images, diff_images

    if type(depth) not in [list, tuple]:
        depth = [depth]
    # from now depth should be a list whose elements are tensors of shape [batch_size x num_cams x hxw]

    total_loss = 0
    warped_image_results = []
    difference_image_results = []
    # for every scale of the scale pyramid compute the reconstruction error
    for d in depth:
        loss, warped, diff = one_scale(d)
        total_loss += loss 
        warped_image_results.append(warped)
        difference_image_results.append(diff)
    
    return total_loss, warped_image_results, difference_image_results

# this function is used by deprecated code
def photometric_reconstruction_loss(
    tgt_img, ref_imgs, intrinsics,
    depth, explainability_mask, poses_est_refs_tgt,
    rotation_mode='euler', padding_mode='zeros'):
    """
    Computes the photometric reconstruction loss as described in Unsupervised Learning of Depth and
    Ego-Motion from Video, Zhou et al.

    :param tgt_img: The target image - this is a single image
    :type tgt_img: tensor or array
    :param ref_imgs: list of reference images
    :type ref_imgs: list
    :param intrinsics: Intrinsics matrix
    :type intrinsics: tensor or array
    :param depth: Depth map of the target frame at multiple scales of the scale pyramid - inverse of output of dispnet
    :type depth: tuple or list
    :param explainability_mask: explainability mask
    :type explainability_mask: array or tensor
    :param poses_est_refs_tgt: poses of the target frame relative to the reference frames. [B x SeqLen-1 x 6]
    :type poses_est_refs_tgt: tensor
    :param rotation_mode: format in which the rotation components of the pose are specified. "euler" or "quat"
    :type rotation_mode: str
    :param padding_mode: padding mode for outside grid values - zeros, border or reflection. Default: zeros
    :type padding_mode: str
    :return: the photometric reconstruction loss, warped reference images, difference between tgt and warped ref images
    :rtype: float, list, list
    """
    def one_scale(depth_tgt, explainability_mask):
        """
        Computes the reconstruction loss between the target_image and all the reference images. This is the sum of the
        mean of the absolute photometric error between the target image and the warped reference images, given a depth
        predition of the the target image. This target depth frame here is at one scale of the scale pyramid. Hence the
        name one_scale.

        :param depth_tgt: depth of the target image
        :type depth_tgt: tensor or array
        :param explainability_mask: Explainability mask
        :type explainability_mask: tensor or array
        :return: the reconstruction loss for
        :rtype: float
        """
        assert(explainability_mask is None or depth_tgt.size()[2:] == explainability_mask.size()[2:])
        assert(poses_est_refs_tgt.size(1) == len(ref_imgs))

        reconstruction_loss = 0
        b, _, h, w = depth_tgt.size()
        downscale = tgt_img.size(2)/h

        # scale the target and reference images to match the size of the depth image by performing interpolation
        tgt_img_scaled = torch.nn.functional.interpolate(tgt_img, (h, w), mode='area')
        ref_imgs_scaled = [torch.nn.functional.interpolate(ref_img, (h, w), mode='area') for ref_img in ref_imgs]
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)

        warped_imgs = []
        diff_maps = []

        for i, ref_img in enumerate(ref_imgs_scaled):
            # pose of the target image relative to the reference images
            current_pose = poses_est_refs_tgt[:, i]
            # warp the reference image to the target frame, and also get the mask of valid points
            ref_img_warped, valid_points = inverse_warp(ref_img, depth_tgt[:,0],
                                                        current_pose, intrinsics_scaled, rotation_mode, padding_mode)
            # compute the difference between the target image and the warped reference image, with mask applied
            diff = (tgt_img_scaled - ref_img_warped) * valid_points.unsqueeze(1).float()
            # if there is an explainability mask, apply that as well
            if explainability_mask is not None:
                diff = diff * explainability_mask[:,i:i+1].expand_as(diff)
            # compute the mean of the absolute difference and add it to the reconstruction loss
            reconstruction_loss += diff.abs().mean()
            assert((reconstruction_loss == reconstruction_loss).item() == 1)

            warped_imgs.append(ref_img_warped[0])
            diff_maps.append(diff[0])

        return reconstruction_loss, warped_imgs, diff_maps

    warped_results, diff_results = [], []
    if type(explainability_mask) not in [tuple, list]:
        explainability_mask = [explainability_mask]
    if type(depth) not in [list, tuple]:
        depth = [depth]

    total_loss = 0
    # for every scale of the scale pyramid compute the reconstruction error
    for d, mask in zip(depth, explainability_mask):
        loss, warped, diff = one_scale(d, mask)
        total_loss += loss
        warped_results.append(warped)
        diff_results.append(diff)
    return total_loss, warped_results, diff_results


def explainability_loss(mask):
    """
    This is the loss introduced in Unsupervised Learning of Depth and Ego-Motion from Video by Tinghui Zhou et al.
    https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/cvpr17_sfm_final.pdf
    :param mask:
    :type mask:
    :return:
    :rtype:
    """
    if type(mask) not in [tuple, list]:
        mask = [mask]
    loss = 0
    for mask_scaled in mask:
        ones_var = torch.ones_like(mask_scaled)
        loss += nn.functional.binary_cross_entropy(mask_scaled, ones_var)
    return loss


def gradient(pred):
    """
    Returns the horizontal and vertical gradients if the input image
    :param pred: input image
    :type pred: array
    :return: difference of pixels in horizontal direction and difference of pixels in vertical direction
    :rtype: tuple
    """
    diff_y = pred[:, :, 1:] - pred[:, :, :-1]
    diff_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    return diff_x, diff_y


def smooth_loss(pred_map):
    """
    Loss that relates to smoothness of the input image (usually depth image).
    This loss is the sum of mean of absolute second order gradients (dI/dx*dI/dx, dI/dy*dI/dy, dI/dx*dI/dy, dI/dy*dI/dy)
    In other words this is the sum of L1 norms of the second-order gradients.
    The intuition is to force the network to predict smooth but not constant depth values.
    This is introduced in https://arxiv.org/pdf/1704.07804v1.pdf. Zhou et al. also use this formulation.

    In Ravi Garg et al. (https://arxiv.org/pdf/1603.04992.pdf), smoothness loss is weighted by a factor of 0.01 when
    added to the total loss. In Zhou et al. this factor is set to 0.5/l, where l is the downscaling factor
    (so for l = 2 the weight is 0.25). We are using a weight of 0.3.

    :param pred_map: input depth image or a list/tuple of depth images moving from finer to coarser resolutions
    :type pred_map: list or array
    :return: loss
    :rtype: tensor
    """

    # convert the input to a list if it is not already one
    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1.

    for scaled_map in pred_map:
        dx, dy = gradient(scaled_map)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
        # we weigh down the loss from the coarser resolution of the depth image
        weight /= 2.3  # don't ask me why it works better
    return loss


def grad_tensor_size(t):
    """
    returns the number of elements in the gradient tensor

    :param t: input tensor
    :type t: tensor
    :return: number of elements in the tensor
    :rtype: int
    """
    return t.size()[1] * t.size()[2] * t.size()[3]


def total_variation_squared_loss(pred_map):
    """
    Computes total variation loss as follows.
    1. Compute grad_x, grad_y
    2. Compute mean grad_x^2 and mean grad_y^2 over the image
    3. Sum them up and multiply by 2
    4. Take the mean over the batch size
    References:  https://github.com/haofeixu/cs231n/blob/master/assignment3/StyleTransfer-TensorFlow.ipynb
    https://github.com/jxgu1016/Total_Variation_Loss.pytorch/blob/master/TVLoss.py
    :param pred_map: input depth image or a list/tuple of depth images moving from finer to coarser resolutions
    :type pred_map: list or array
    :return: loss
    :rtype: float
    """
    # convert the input to a list if it is not already one
    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1.0
    for scaled_map in pred_map:
        batch_size = scaled_map.size()[0]
        dx, dy = gradient(scaled_map)
        count_x = grad_tensor_size(scaled_map[:,:,1:,:])
        count_y = grad_tensor_size(scaled_map[:, :, :, 1:])
        loss += weight * 2 * (torch.pow(dx, 2).sum() / count_x + torch.pow(dy, 2).sum() / count_y) / batch_size
        weight /= 2.0   # reduce the weight by 2 for the next level of the scale pyramid
    return loss


def total_variation_loss(pred_map, sum_or_mean="sum"):
    """
    Computes total variation loss as follows.
    1. Compute absolute values of grad_x, grad_y
    2. Sum them up and multiply by 2
    3. Take the mean over the batch size
    References: https://en.wikipedia.org/wiki/Total_variation_denoising
    https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/ops/image_ops_impl.py#L3147-L3154
    :param pred_map: input depth image or a list/tuple of depth images moving from finer to coarser resolutions
    :type pred_map: list or array
    :return: loss
    :rtype: float
    """
    # convert the input to a list if it is not already one
    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1.0
    for scaled_map in pred_map:
        batch_size = scaled_map.size()[0]
        dx, dy = gradient(scaled_map)
        if sum_or_mean == "sum":
            loss += weight * (torch.abs(dx).sum() + torch.abs(dy).sum()) / batch_size
        elif sum_or_mean == "mean":
            count_x = grad_tensor_size(scaled_map[:, :, 1:, :])
            count_y = grad_tensor_size(scaled_map[:, :, :, 1:])
            loss += weight * (torch.abs(dx).sum() / count_x + torch.abs(dy).sum() / count_y) / batch_size
        else:
            raise ValueError("sum_or_mean should be sum or mean")
        weight /= 2.0   # reduce the weight by 2 for the next level of the scale pyramid
    return loss


def pose_loss(pose, pose_gt):
    """
    Computes the mean distance (in metres) between the poses and the mean relative angle (in radians) between the poses

    :param pose: predicted pose in [x, y, z, axang format]
    :type pose: tensor
    :param pose_gt: ground truth pose as a 4x4 matrix
    :type pose_gt: tensor
    :return: mean_distance_error, mean angle error
    :rtype: tensor, tensor
    """
    # This is what joe was doing and is crap
    # pred_pose_magnitude = pose[:, :, :3].norm(dim=2)
    # pose_gt_magnitude = pose_gt[:, :, :3].norm(dim=2)
    # error = (pred_pose_magnitude - pose_gt_magnitude).abs().mean()
    # return error

    batch_size = pose.size(0)
    ref_images =  pose.size(1)
    pose_est = utils.get_4x4_from_pose_tensor(pose, "axang")

    # mean distance error is the mean of the norm of the difference in translation components
    diff = pose_est[:, :, 0:3, -1:] - pose_gt[:, :, 0:3, -1:]     # difference between translations
    mean_distance_error = diff.norm(p=2, dim=1).mean()      # take mean of the norm of the difference

    # mean angle error is the mean of the angle (radians) computed from the relative rotation, i.e. R_est.inv() x R_gt
    rot_est_inv = torch.transpose(pose_est[:, :, 0:3, 0:3], 2, 3)
    relative_rotation = torch.matmul(rot_est_inv, pose_gt[:, :, 0:3, 0:3])

    mean_angle_error = torch.zeros(1, device=pose.device)
    minus_one = torch.ones(1, device=pose.device) * -1  # -1 tensor
    plus_one = torch.ones(1, device=pose.device)        # 1 tensor
    # compute angle from the rotation matrix, [1 + 2cos(angle) = trace of the matrix]
    for b in range(0, batch_size):
        for r in range(0, ref_images):
            tr = (relative_rotation[b, r, 0:3, 0:3].trace() - 1) / 2
            mean_angle_error += torch.acos(torch.min(plus_one, torch.max(minus_one, tr))) # angle in the range of 0 to pi
    mean_angle_error /= (batch_size * ref_images)      # compute mean angle

    return mean_distance_error, mean_angle_error


def forward_backward_loss(pose_1_2, pose_2_1):
    batch_size = pose_1_2.size(0)
    ref_images = pose_1_2.size(1)

    pose_1_2_mat = utils.get_4x4_from_pose_tensor(pose_1_2, "axang")
    pose_2_1_mat = utils.get_4x4_from_pose_tensor(pose_2_1, "axang")

    error = torch.zeros(1, device=pose_1_2.device)
    for b in range(0, batch_size):
        for r in range(0, ref_images):
            diff = torch.matmul(torch.inverse(pose_1_2_mat[b, r, :, :]), pose_2_1_mat[b, r, :, :])
            error += torch.sum(torch.abs(torch.eye(4, device=pose_1_2.device) - diff))
    error / (batch_size*ref_images)
    return error


@torch.no_grad()
def compute_errors(gt, pred, crop=True):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0,0,0,0,0,0
    batch_size = gt.size(0)

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if crop:
        crop_mask = gt[0] != gt[0]
        y1,y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1,x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2,x1:x2] = 1

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0) & (current_gt < 80)
        if crop:
            valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, 80)

        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]
