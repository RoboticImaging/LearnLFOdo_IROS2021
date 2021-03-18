from __future__ import division
import torch
import random
import numpy as np
import math
import cv2

'''Set of transform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''


class Compose(object):
    """
    Transform that composes transforms.
    """
    def __init__(self, transforms):
        """
        Initializer

        :param transforms: list of transforms
        :type transforms: list
        """
        self.transforms = transforms

    def __call__(self, images, intrinsics):
        """
        Call to the transform.

        :param images: list of image tensors on which the transforms have to be applied
        :type images: list
        :param intrinsics: intrinsics tensor
        :type intrinsics: tensor
        :return: images with all the transforms applied on it and the intrinsics tensor
        :rtype: tuple
        """
        for t in self.transforms:
            images, intrinsics = t(images, intrinsics)
        return images, intrinsics


class Normalize(object):
    """
    Normalizes an image tensor with the specified mean and standard deviation.
    """
    def __init__(self, mean, std):
        """
        Initializes the transform.

        :param mean: mean of the tensor data
        :type mean: float
        :param std: standard deviation of the tensor data
        :type std: float
        """
        self.mean = mean
        self.std = std

    def __call__(self, images, intrinsics):
        """
        Call to the transform. Normalizes the input tensor.

        :param images: list of image tensors that have to be normalized
        :type images: list
        :param intrinsics: Intrinsics tensor. Not used, just returned as is
        :type intrinsics: tensor
        :return: normalized image tensors and the intrinsics tensor
        :rtype: tuple
        """
        for tensor in images:
            tensor.sub_(self.mean).div_(self.std)
        return images, intrinsics


class ArrayToTensor(object):
    """
    Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list
    of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor.
    """

    def __call__(self, images, intrinsics):
        """
        Call to the transform.

        :param images: list of numpy.ndarray of shape (H x W x C)
        :type images: list
        :param intrinsics: intrinsics matrix
        :type intrinsics: numpy array
        :return: list of image tensors of shape (C x H x W) and the intrinsics tensor
        :rtype: tuple
        """
        tensors = []
        for im in images:
            # put it from HWC to CHW format
            if (len(im.shape)) > 2:
                im = np.transpose(im, (2, 0, 1))
            else:
                im = im.reshape([1, im.shape[0], im.shape[1]])
            # handle numpy array
            tensors.append(torch.from_numpy(im).float() / 255.0)
        return tensors, intrinsics


class RandomHorizontalFlip(object):
    """
    Randomly horizontally flips the given numpy array with a probability of 0.5
    """

    def __call__(self, images, intrinsics):
        """
        Call to the transform

        :param images: list of image tensors
        :type images: list
        :param intrinsics: intrinsics tensor. not used. returned as is.
        :type intrinsics: tensor
        :return: list of transformed images and the intrinsics tensor
        :rtype: list
        """
        assert intrinsics is not None
        if random.random() < 0.5:
            output_intrinsics = np.copy(intrinsics)
            output_images = [np.copy(np.fliplr(im)) for im in images]
            w = output_images[0].shape[1]
            output_intrinsics[0, 2] = w - output_intrinsics[0, 2]
        else:
            output_images = images
            output_intrinsics = intrinsics
        return output_images, output_intrinsics


class RandomScaleCrop(object):
    """
    Randomly zooms images up to 15% and crops them to keep same size as before.
    """

    def __call__(self, images, intrinsics):
        """
        Call to the transform

        :param images: list of image tensors
        :type images: list
        :param intrinsics: intrinsics tensor. not used. returned as is.
        :type intrinsics: tensor
        :return: list of transformed images and the intrinsics tensor
        :rtype: list
        """
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)

        in_h, in_w, _ = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1, 1.15, 2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

        output_intrinsics[0] *= x_scaling
        output_intrinsics[1] *= y_scaling

        # scaled_images = [imresize(im, (scaled_h, scaled_w)) for im in images]   # scipy imresize is Deprecated
        scaled_images = [cv2.resize(im, (scaled_w, scaled_h)) for im in images]  # Note width first and height next

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        cropped_images = [im[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for im in scaled_images]

        output_intrinsics[0, 2] -= offset_x
        output_intrinsics[1, 2] -= offset_y

        return cropped_images, output_intrinsics
