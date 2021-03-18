import os
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data
# from imageio import imread
# from path import Path

from utils import get_relative_6dof
from epimodule import load_multiplane_focalstack


def load_as_float(path, gray=False):
    """
    Loads an image as a float and returns it as a numpy array.

    :param path: Path to the image file.
    :type path: str
    :param gray: Boolean indicating if the image should be converted to grayscale. Default is False.
    :type gray: bool
    :return: The image as a numpy array
    :rtype: numpy array
    """
    # im = imread(path).astype(np.float32)  # Switched to opencv image loader
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    if gray:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return im.astype(np.float32)


def load_lightfield(path, cameras, gray):
    """
    Loads a lightfield image from a set of cameras and returns a list of images

    :param path: Path to the images
    :type path: str
    :param cameras: List of cameras
    :type cameras: list
    :param gray: Boolean indicating if the images should be converted to grayscale
    :type gray: bool
    :return: List of images that constitute the lightfield image
    :rtype: list
    """

    imgs = []
    for cam in cameras:
        imgpath = path.replace('/8/', '/{}/'.format(cam))
        imgs.append(load_as_float(imgpath, gray))

    return imgs


def load_relative_pose(tgt, ref):
    """
    Loads relative poses between the target and reference image

    :param tgt: The name of the target image
    :type tgt: str
    :param ref: The name of the reference image
    :type ref: str
    :return: The relative pose from target to reference image.
    :rtype: Array
    """

    # Get the number in the filename - super hacky
    sequence_name = os.path.join("/", *tgt.split("/")[:-2])
    tgt_id = int(tgt.split("/")[-1].split(".")[-2])
    ref_id = int(ref.split("/")[-1].split(".")[-2])
    # pose_file = np.load(os.path.join(sequence_name, "poses_gt_absolute.npy"))
    pose_file = np.load(os.path.join(sequence_name, "poses_gt_base_cam.npy"))
    tgt_pose = pose_file[tgt_id, :]    # this is a 4x4 matrix
    ref_pose = pose_file[ref_id, :]    # this is a 4x4 matrix
    # rel_pose = get_relative_6dof(tgt_pose[:3], tgt_pose[3:], ref_pose[:3], ref_pose[3:], rotation_mode='euler')
    rel_pose = get_relative_6dof(tgt_pose[0:3, 3], tgt_pose[0:3, 0:3],
                                 ref_pose[0:3, 3], ref_pose[0:3, 0:3],
                                 rotation_mode='rotm', return_as_mat=True)
    return rel_pose


class SequenceFolder(data.Dataset):
    """
    Class for loading and processing data stored in a sequential way.

    The folder structure should be as follows:\n
    - root/scene_1/0000000.jpg
    - root/scene_1/0000001.jpg
    - ..
    - root/scene_1/cam.txt
    - root/scene_2/0000000.jpg

    For every scene, we have a sequence of images and the camera intrinsics matrix.

        Can load images as focal stack, must pass in arguments lf_format='focalstack', num_cameras, num_planes.
    """

    def __init__(self,
                 root,
                 cameras=[8],
                 gray=False,
                 seed=None,
                 train=True,
                 sequence_length=3,
                 transform=None,
                 target_transform=None,
                 shuffle=True,
                 sequence=None,
                 lf_format='stack',  # Parameters to change if using focal stack only
                 num_cameras=None,  # ========
                 num_planes=None  # ========
                 ):
        """
        Initialization function

        :param root: Path to the folder where the data is present.
        This folder should have subfolders seq#, train.txt, val.txt
        :type root: str
        :param cameras: List of camera indices to use when creating the light field image stack
        :type cameras: list
        :param gray: Boolean indicating if the images have to be converted to grayscale
        :type gray: bool
        :param seed: Seed for random number generator
        :type seed: int
        :param train: Boolean indicating if the data to be loaded is the training or the validation set. Default is True
        :type train: bool
        :param sequence_length: Number of images in the sequence. Default 3
        :type sequence_length: int
        :param transform: Transform that has to be applied to the images. Default is None
        :type transform: Tensor
        :param target_transform: [Unused] Another transform, probably for target images?
        :type target_transform: Tensor
        :param shuffle: Boolean indicating if the data has to be shuffled. Default is True for training.
        :type shuffle: bool
        :param sequence: Sequence index that has to be loaded, if required. Default is None to load all data.
        :type sequence: int
        :param lf_format: Format of the light field image - "stack" or "focalstack"
        :type lf_format: str
        :param num_cameras: Number of cameras
        :type num_cameras: int
        :param num_planes: Number of image planes in the multi-plane focal stack
        :type num_planes: int
        """

        assert lf_format in ["stack", "focalstack"]

        np.random.seed(seed)
        random.seed(seed)

        self.cameras = cameras                      # indices of cameras to use for forming the light field image
        self.gray = gray                            # boolean indicating if the image should be converted to grayscale
        self.root = root                            # root directory where the data is present
        self.shuffle = shuffle                      # boolean indicating if data has to be shuffled
        self.lf_format = lf_format                  # format of the light field image
        self.num_cameras = num_cameras              # number of cameras
        self.num_planes = num_planes                # number of image planes in the multi-plane focal stack
        self.sequence_length = sequence_length      # number of images in a sequence
        self.samples = None                         # the data samples
        self.transform = transform                  # transform to apply on the images
        self.intrinsics_file = "../intrinsics.txt"  # file where the camera intrinsics are stored

        scene_list_path = self.root + '/train.txt' if train else self.root + '/val.txt'

        # Hardcoding to load image from central camera (8)
        if sequence is not None:
            # choose the particular scene (called seq# in the folder)
            self.scenes = [self.root + "/seq" + str(sequence) + "/8"]
        else:
            # get all the scenes (called seq# in the folder) from the the list of scenes.
            # Note: Need to strip newline first
            self.scenes = [self.root + "/" + folder[:].rstrip() + "/8" for folder in open(scene_list_path)]

        self.crawl_folders()

    def crawl_folders(self):
        """
        Crawls through the folders that represent a scene and creates a list of images to load.
        This is stored in the member variable self.samples
        """
        sequence_set = []
        demi_length = (self.sequence_length - 1) // 2           # floor( (sequence_length-1) /2)
        shifts = list(range(-demi_length, demi_length + 1))     # create list [-d, -(d-1), ..-1, 0, 1, ..., d-1, d]
        shifts.pop(demi_length)                                 # remove element 0 from the list

        for scene in self.scenes:
            # load intrinsics matrix from a text file. This file is present in the same folder as this script
            intrinsics = np.genfromtxt(self.intrinsics_file).astype(np.float32).reshape((3, 3))

            # load image files - replaced the use of module path.files with os.walk
            # imgs = sorted(scene.files('*.png'))
            imgs = []
            for _, _, files in os.walk(scene):
                for img_file in files:
                    if img_file.endswith(".png"):
                        imgs.append(img_file)
            imgs.sort()

            if len(imgs) < self.sequence_length:
                continue

            # form a sequence set. It contains the following
            # intrinsics matrix,
            # target image = image at index i
            # reference images = list of images at indices i-demi_length to i+demi_length except image at i
            for i in range(demi_length, len(imgs) - demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i + j])
                sequence_set.append(sample)

        if self.shuffle:
            random.shuffle(sequence_set)

        self.samples = sequence_set

    def __getitem__(self, index):
        """
        Returns the element at the desired index from the dataset.

        :param index: Index of the element of the dataset
        :type index: int
        :return: target image, target light field, reference images, reference light fields, intrinsics,
        inverse of the intrinsics, pose
        :rtype: tuple
        """
        sample = self.samples[index]
        # load target and reference images
        tgt_img = load_as_float(sample['tgt'], False)
        ref_imgs = [load_as_float(ref_img, False) for ref_img in sample['ref_imgs']]

        # load target and reference lightfields
        tgt_lf = None
        ref_lfs = None
        if self.lf_format == 'stack':
            tgt_lf = load_lightfield(sample['tgt'], self.cameras, self.gray)
            ref_lfs = [load_lightfield(ref_img, self.cameras, self.gray) for ref_img in sample['ref_imgs']]

        # load target and reference multiplane focalstacks
        elif self.lf_format == 'focalstack':
            tgt_lf = load_multiplane_focalstack(sample['tgt'],
                                                num_planes=self.num_planes,
                                                num_cameras=self.num_cameras,
                                                gray=self.gray)
            ref_lfs = [load_multiplane_focalstack(ref_img,
                                                  num_planes=self.num_planes,
                                                  num_cameras=self.num_cameras,
                                                  gray=self.gray) for ref_img in sample['ref_imgs']]

        # load ground truth relative pose - this is a 4x4 matrix
        pose = torch.Tensor([load_relative_pose(sample['tgt'], ref) for ref in sample['ref_imgs']])

        # apply transforms if present
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))

            # Lazy reuse of existing function
            tgt_lf, _ = self.transform(tgt_lf, np.zeros((3, 3)))
            ref_lfs = [self.transform(ref, np.zeros((3, 3)))[0] for ref in ref_lfs]

            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])

        tgt_lf = torch.cat(tgt_lf, 0)  # Concatenate lightfield on colour channel
        ref_lfs = [torch.cat(ref, 0) for ref in ref_lfs]

        return tgt_img, tgt_lf, ref_imgs, ref_lfs, intrinsics, np.linalg.inv(intrinsics), pose

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        :return: Number of samples in the dataset
        :rtype: int
        """
        return len(self.samples)


def get_focal_stack_loaders(args, train_transform, valid_transform, shuffle=True):
    """
    Returns the dataset objects with images as a focal stack.

    :param args: user specified arguments
    :type args: argparse object
    :param train_transform: Transform to apply on the training images
    :type train_transform: Tensor
    :param valid_transform: Transform to apply on the validation images
    :type valid_transform: Tensor
    :param shuffle: Boolean indicating if data has to be shuffled
    :type shuffle: bool
    :return: Training and Validation datasets
    :rtype: tuple
    """
    train_set = SequenceFolder(
        args.data,
        gray=args.gray,
        transform=train_transform,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length,
        lf_format='focalstack',
        num_cameras=args.num_cameras,
        num_planes=args.num_planes,
        shuffle=shuffle
    )

    val_set = SequenceFolder(
        args.data,
        gray=args.gray,
        transform=valid_transform,
        seed=args.seed,
        train=False,
        sequence_length=args.sequence_length,
        lf_format='focalstack',
        num_cameras=args.num_cameras,
        num_planes=args.num_planes,
        shuffle=False       # shuffle is false for validation
    )

    return train_set, val_set


def get_validation_focal_stack_loader(args, sequence=None, transform=None):
    """
    Returns the dataset objects with images as a focal stack for validation

    :param args:user specified arguments
    :type args: argparse object
    :param sequence: Index of the sequence to use for validation
    :type sequence: int
    :param transform: Transform to apply on the images
    :type transform: Tensor
    :return: Validation dataset
    :rtype:
    """
    return SequenceFolder(
        args.data,
        gray=args.gray,
        transform=transform,
        seed=args.seed,
        train=False,
        sequence_length=args.sequence_length,
        lf_format='focalstack',
        num_cameras=args.num_cameras,
        num_planes=args.num_planes,
        shuffle=False,       # shuffle is false for validation
        sequence=sequence
    )


def get_stacked_lf_loaders(args, train_transform, valid_transform, shuffle=True):
    """
    Returns the dataset objects with light field image stacks.

    :param args: user specified arguments
    :type args: argparse object
    :param train_transform: Transform to apply on the training images
    :type train_transform: Tensor
    :param valid_transform: Transform to apply on the validation images
    :type valid_transform: Tensor
    :param shuffle: Boolean indicating if data has to be shuffled
    :type shuffle: bool
    :return: Training and Validation datasets
    :rtype: tuple
    """

    train_set = SequenceFolder(
        args.data,
        gray=args.gray,
        cameras=args.cameras,
        transform=train_transform,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length,
        lf_format='stack',
        shuffle=shuffle
    )

    val_set = SequenceFolder(
        args.data,
        gray=args.gray,
        cameras=args.cameras,
        transform=valid_transform,
        seed=args.seed,
        train=False,
        sequence_length=args.sequence_length,
        lf_format='stack',
        shuffle=False       # shuffle is false for validation
    )

    return train_set, val_set


def get_validation_stacked_lf_loader(args, sequence=None, transform=None):
    """
    Returns the dataset objects with images as light field image stacks for validation

    :param args:user specified arguments
    :type args: argparse object
    :param sequence: Index of the sequence to use for validation
    :type sequence: int
    :param transform: Transform to apply on the images
    :type transform: Tensor
    :return: Validation dataset
    :rtype:
    """
    return SequenceFolder(
        args.data,
        gray=args.gray,
        cameras=args.cameras,
        transform=transform,
        seed=args.seed,
        train=False,
        sequence_length=args.sequence_length,
        lf_format='stack',
        shuffle=False,       # shuffle is false for validation
        sequence=sequence
    )
