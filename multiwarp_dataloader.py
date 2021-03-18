import torch.utils.data as data
import numpy as np
import random
import torch
import os

from epimodule import load_multiplane_focalstack
from epimodule import load_tiled_epi_vertical, load_tiled_epi_horizontal, load_tiled_epi_full
from epimodule import load_stacked_epi, load_stacked_epi_no_repeats
from epimodule import load_lightfield, load_relative_pose
from epimodule import DEFAULT_PATCH_INTRINSICS


class MetaData:
    """
    Storage class for metadata that might be needed during evaluation

    """
    def __init__(self, cameras, tgt_name, ref_names, gray, flipped):
        """
        Initializes a dictionary of metadata.

        :param cameras: List of camera indices
        :type cameras: list
        :param tgt_name: Filename of target image
        :type tgt_name: str
        :param ref_names: List of file names of reference images
        :type ref_names: list
        :param gray: Boolean indicating grayscale or not
        :type gray: bool
        :param flipped: Boolean indicating if the image is flipped or not
        :type flipped: bool
        """
        self.metadata = {
            "cameras": cameras,                 # List of camera indices
            "tgt_name": tgt_name,               # Filename
            "ref_names": ref_names,             # Filenames
            "gray": gray,                       # Is grayscale
            "flipped": flipped,                 # Is flipped
        }

    def get_as_dict(self):
        """
        Returns the metadata as a dictionary.

        :return: Dictionary of metadata
        :rtype: dict
        """
        return self.metadata


class TrainingData:
    """
    Storage class for data that is needed during training.
    The training routine expects a dict with the following fields.
    This class ensures a consistent access API for different data loading modules.
    The __getitem__ method of any dataset class should create one of these, and
    return the dictionary obtained from TrainingData.get_as_dict()

    """
    def __init__(self, tgt, tgt_formatted, ref, ref_formatted, intrinsics, poses_gt_refs_tgt, metadata):
        """
        Initializer

        :param tgt: List of tgt frame lightfield sub-apertures for multi-warp photometric loss
        :type tgt: tensor
        :param tgt_formatted: The encoded target lightfield, to be input to the disp and pose networks
        :type tgt_formatted: tensor
        :param ref: List of list of ref frame lightfield sub-apertures for multi-warp photometric loss
        :type ref: tensor
        :param ref_formatted: List of encoded reference lightfields, to be input to the disp and pose networks
        :type ref_formatted: list
        :param intrinsics: Camera intrinsics matrix K
        :type intrinsics: numpy.array
        :param poses_gt_refs_tgt: Ground truth poses of the target frame wrt the reference frames - [refs x 4x4 matrix]
        :type poses_gt_refs_tgt: tensor
        :param metadata: Metadata. Not used in training but only in evaluation.
        :type metadata: MetaData
        """
        self.training_data = {
            "tgt_lf": tgt,      # list of tgt frame lightfield sub-apertures for multi-warp photometric loss
            "ref_lfs": ref,     # list of list of ref frame lightfield sub-apertures for multi-warp photometric loss
            "tgt_lf_formatted": tgt_formatted,                  # The encoded tgt lightfield image
            "ref_lfs_formatted": ref_formatted,                 # The list of encoded ref lightfield images
            "poses_gt_refs_tgt": poses_gt_refs_tgt,             # Ground truth poses of the ref frames in the tgt frame
            "metadata": metadata.get_as_dict(),                 # Metadata (not used for training but for eval)
            "intrinsics": intrinsics,                           # Intrinsics K
            "intrinsics_inv": np.linalg.inv(intrinsics)         # Intrinsics^-1 TODO: why pass this?
        }

    def get_as_dict(self):
        """
        Returns the training data as a dictionary
        :return: The dictionary of training data
        :rtype: dict
        """
        return self.training_data


class TrainingDataEpi:
    """
    Storage class for data that is needed during training.
    The training routine expects a dict with the following fields.
    This class ensures a consistent access API for different data loading modules.
    The __getitem__ method of any dataset class should create one of these, and
    return the dictionary obtained from TrainingData.get_as_dict()
    """

    def __init__(self, tgt, tgt_formatted_h, tgt_formatted_v, ref, ref_formatted_h, ref_formatted_v,
                 tgt_stack, ref_stacks, intrinsics, poses_gt_refs_tgt, metadata):
        """
        Initializer

        :param tgt: List of tgt frame lightfield sub-apertures for multi-warp photometric loss
        :type tgt: tensor
        :param tgt_formatted_h: The encoded tgt lightfield image for horizontal sub-apertures
        :type tgt_formatted_h: tensor
        :param tgt_formatted_v: The encoded tgt lightfield image for vertical sub-apertures
        :type tgt_formatted_v: tensor
        :param ref: List of list of ref frame lightfield sub-apertures for multi-warp photometric loss
        :type ref: tensor
        :param ref_formatted_h: List of encoded ref lightfield images of horizontal sub-apertures
        :type ref_formatted_h: tensor
        :param ref_formatted_v: List of encoded ref lightfield images of vertical sub-apertures
        :type ref_formatted_v: tensor
        :param tgt_stack: List of tgt lightfield sub-apertures for sending as stack to encoders
        :type tgt_stack: tensor
        :param ref_stacks: List of list of ref lightfield sub-apertures for sending as stack to encoders
        :type ref_stacks: tensor
        :param intrinsics: Camera intrinsics matrix K
        :type intrinsics: ndarray
        :param poses_gt_refs_tgt: Ground truth poses of the target frame wrt the reference frames - [refs x 4x4 matrix]
        :type poses_gt_refs_tgt: tensor
        :param metadata: Metadata. Not used in training but only in evaluation.
        :type metadata: MetaData
        """

        self.training_data = {
            "tgt_lf": tgt,  # list of tgt frame lightfield sub-apertures for multi-warp photometric loss
            "ref_lfs": ref,  # list of list of ref frame lightfield sub-apertures for multi-warp photometric loss
            "tgt_lf_formatted_h": tgt_formatted_h,  # The encoded tgt lightfield image for horizontal sub-apertures
            "tgt_lf_formatted_v": tgt_formatted_v,  # The encoded tgt lightfield image for vertical sub-apertures
            "ref_lfs_formatted_h": ref_formatted_h,  # List of encoded ref lightfield images of horizontal sub-apertures
            "ref_lfs_formatted_v": ref_formatted_v,  # List of encoded ref lightfield images of vertical sub-apertures
            "tgt_stack": tgt_stack,    # list of tgt lightfield sub-apertures for sending as stack to encoders
            "ref_stacks": ref_stacks,  # list of list of ref lightfield sub-apertures for sending as stack to encoders
            "poses_gt_refs_tgt": poses_gt_refs_tgt,  # List of ground truth poses of the tgt frame wrt ref frames
            "metadata": metadata.get_as_dict(),  # Metadata (not used for training but for eval)
            "intrinsics": intrinsics,  # Intrinsics K
            "intrinsics_inv": np.linalg.inv(intrinsics)  # Intrinsics^-1 TODO: why pass this?
        }

    def get_as_dict(self):
        """
        Returns the training data as a dictionary
        :return: The dictionary of training data
        :rtype: dict
        """
        return self.training_data


class BaseDataset(data.Dataset):
    """
    Base class for loading epi-module data-sets. Takes care of crawling the root directory and storing some common
    configuration parameters.
    """

    def __init__(self, root, cameras, gray, seed, train, sequence_length, transform, shuffle, sequence):
        """
        Initializer

        :param root: Path to the folder where the data is present.
        This folder should have subfolders seq#, train.txt, val.txt
        :type root: str
        :param cameras: List of camera indices to use for computing the multi-warp photometric error
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
        :type transform: tensor
        :param shuffle: Boolean indicating if the data has to be shuffled. Default is True for training.
        :type shuffle: bool
        :param sequence: Sequence index that has to be loaded, if required. Default is None to load all data.
        :type sequence: int
        """
        np.random.seed(seed)
        random.seed(seed)

        self.samples = None         # the data samples
        self.cameras = cameras      # indices of cameras to use for computing the multi-warp photometric error
        self.gray = gray            # boolean indicating if the image should be converted to grayscale
        self.root = root            # root directory where the data is present
        self.shuffle = shuffle      # boolean indicating if data has to be shuffled
        self.transform = transform              # transform to apply on the images
        # number of images in a sequence (reference images + target image),
        # if sequencce_length = 2 then reference image index is target-1
        self.sequence_length = sequence_length

        scene_list_path = self.root + '/train.txt' if train else self.root + '/val.txt'

        # Hardcoding to load image from central camera (8)
        if sequence is not None:
            # choose the particular scene (called seq# in the folder)
            self.scenes = [self.root + "/seq" + str(sequence) + '/8']
        else:
            # get all the scenes (called seq# in the folder) from the the list of scenes.
            # Note: Need to strip newline first
            self.scenes = [self.root + "/" + folder[:].rstrip() + '/8' for folder in open(scene_list_path)]

        self.crawl_folders()

    def crawl_folders(self):
        """
        Crawls through the folders that represent a scene and creates a list of images to load.
        This is stored in the member variable self.samples and consists of the following
        the intrinsics matrix,
        target image = image at index i
        reference images = list of images at indices i-demi_length to i+demi_length except image at i
        If sequence_length == 2 then reference images = image at i-1
        """
        sequence_set = []

        if self.sequence_length == 2:
            demi_length = 1
            shifts = [-1]
        else:
            demi_length = (self.sequence_length - 1) // 2           # floor( (sequence_length-1) /2)
            shifts = list(range(-demi_length, demi_length + 1))     # create list [-d, -(d-1), ..-1, 0, 1, ..., d-1, d]
            shifts.pop(demi_length)                                 # remove element 0 from the list

        for scene in self.scenes:
            intrinsics = DEFAULT_PATCH_INTRINSICS

            # load image files - replaced the use of module path.files with os.walk
            # imgs = sorted(scene.files('*.png'))
            imgs = []
            for _, _, files in os.walk(scene):
                for img_file in files:
                    if img_file.endswith(".png"):
                        imgs.append(os.path.join(scene, img_file))
            imgs.sort()

            if len(imgs) < self.sequence_length:
                continue

            # form a sequence set. It contains the following
            # intrinsics matrix,
            # target image = image at index i
            # reference images = list of images at indices i-demi_length to i+demi_length except image at i
            # If sequence_length == 2 then reference images = image at i-1
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
        This function has to be overwritten by children classes that inherit this class.

        :param index: index of the element of the dataset
        :type index: int
        :return: Error
        :rtype: BaseException
        """
        raise NotImplementedError

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        :return: Number of samples in the dataset
        :rtype: int
        """
        return len(self.samples)


class FocalstackLoader(BaseDataset):
    """
    Class for loading lightfield images as a focal stack
    """

    def __init__(self, root, cameras, fs_num_cameras, fs_num_planes, gray=False, seed=None, train=True,
                 sequence_length=3, transform=None, shuffle=True, sequence=None, no_pose=False):
        """
        Initializer

        :param root: Path to the folder where the data is present.
        This folder should have subfolders seq#, train.txt, val.txt
        :type root: str
        :param cameras: List of camera indices to use for computing the multi-warp photometric error.
        :type cameras: list
        :param fs_num_cameras: Number of cameras to use to form the focal stack encoding
        :type fs_num_cameras: int
        :param fs_num_planes: Number of planes in the multi-plane focal stack
        :type fs_num_planes: int
        :param gray: Boolean indicating if the images have to be converted to grayscale
        :type gray: bool
        :param seed: Seed for random number generator
        :type seed: int
        :param train: Boolean indicating if the data to be loaded is the training or the validation set. Default is True
        :type train: bool
        :param sequence_length: Number of images in the sequence. Default 3
        :type sequence_length: int
        :param transform: Transform that has to be applied to the images. Default is None
        :type transform: tensor
        :param shuffle: Boolean indicating if the data has to be shuffled. Default is True for training.
        :type shuffle: bool
        :param sequence: Sequence index that has to be loaded, if required. Default is None to load all data.
        :type sequence: int
        """

        super(FocalstackLoader, self).__init__(root, cameras, gray, seed, train, sequence_length, 
                                               transform, shuffle, sequence)

        self.num_fs_cameras = fs_num_cameras
        self.num_planes = fs_num_planes
        self.no_pose = no_pose

    def __getitem__(self, index):
        """
        Returns the element at the desired index from the dataset.

        :param index: Index of the element of the dataset
        :type index: int
        :return: Dictionary of training data
        :rtype: dict
        """

        # get the data sample at the specified index
        sample = self.samples[index]

        # load target and reference images from teh chosen cameras. This is used only to compute the multi-warp
        # photometric loss. If self.cameras is [8] then single-warp photometric loss is computed.
        tgt_lf = load_lightfield(sample['tgt'], self.cameras, self.gray)
        ref_lfs = [load_lightfield(ref_img, self.cameras, self.gray) for ref_img in sample['ref_imgs']]

        # load target and reference lightfields as a focal stack
        tgt_focalstack = load_multiplane_focalstack(sample['tgt'],
                                                    num_planes=self.num_planes,
                                                    num_cameras=self.num_fs_cameras,
                                                    gray=self.gray)
        
        ref_focalstacks = [load_multiplane_focalstack(ref_img,
                                                      num_planes=self.num_planes,
                                                      num_cameras=self.num_fs_cameras,
                                                      gray=self.gray) for ref_img in sample['ref_imgs']]
        # load ground truth pose of target frame wrt reference frames - [refs x 4x4 matrix]
        poses_gt_refs_tgt = torch.Tensor([load_relative_pose(ref, sample['tgt'], self.no_pose)
                                          for ref in sample['ref_imgs']])
        # intrinsics = np.copy(sample['intrinsics'])

        # apply transforms if present
        if self.transform is not None:
            tgt_lf, _ = self.transform(tgt_lf, np.zeros((3, 3)))
            ref_lfs = [self.transform(ref, np.zeros((3, 3)))[0] for ref in ref_lfs]
            tgt_focalstack, _ = self.transform(tgt_focalstack, np.zeros((3, 3)))
            ref_focalstacks = [self.transform(ref, np.zeros((3, 3)))[0] for ref in ref_focalstacks]

        # concatenate light fields on colour channel
        tgt_lf = torch.cat(tuple(tgt_lf), 0)
        ref_lfs = [torch.cat(ref, 0) for ref in ref_lfs]
        # concatenate multiple focal plane images on colour channel
        tgt_focalstack = torch.cat(tgt_focalstack, 0)
        ref_focalstacks = [torch.cat(ref, 0) for ref in ref_focalstacks]

        # Create metadata and training data
        metadata = MetaData(self.cameras, sample['tgt'], sample['ref_imgs'], self.gray, False)
        trainingdata = TrainingData(tgt_lf, tgt_focalstack,
                                    ref_lfs, ref_focalstacks,
                                    sample['intrinsics'], poses_gt_refs_tgt, metadata)

        return trainingdata.get_as_dict()


class StackedLFLoader(BaseDataset):
    """
    Class for loading lightfield images
    """

    def __init__(self, root, cameras, gray=False, seed=None, train=True, sequence_length=3,
                 transform=None, shuffle=True, sequence=None, cameras_stacked='input',
                 no_pose=False, is_monocular=False):
        """
        Initlializer

        :param root: Path to the folder where the data is present.
        This folder should have subfolders seq#, train.txt, val.txt
        :type root: str
        :param cameras: List of camera indices to use for computing the multi-warp photometric error.
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
        :type transform: tensor
        :param shuffle: Boolean indicating if the data has to be shuffled. Default is True for training.
        :type shuffle: bool
        :param sequence: Sequence index that has to be loaded, if required. Default is None to load all data.
        :type sequence: int
        :param cameras_stacked: Either 'input' or 'full'. If 'input', then the list @cameras is used to generate the
        stack. If 'full' then all the 17 cameras are used to generate the stack.
        :type cameras_stacked: str
        """
        super(StackedLFLoader, self).__init__(root, cameras, gray, seed, train, sequence_length, 
                                              transform, shuffle, sequence)
        assert cameras_stacked in ['input', 'full']
        self.cameras_stacked = cameras_stacked
        if is_monocular:
            self.cameras_stacked_indices = [8]
        else:
            self.cameras_stacked_indices = [3, 8, 13, 7, 9]

        self.no_pose = no_pose

    def __getitem__(self, index):
        """
        Returns the element at the desired index from the dataset.

        :param index: Index of the element of the dataset
        :type index: int
        :return: Dictionary of training data
        :rtype: dict
        """

        # get the data sample at the specified index
        sample = self.samples[index]
        # load target and reference lightfield images images
        # --> list of images that form the target lightfield image
        tgt_lf = load_lightfield(sample['tgt'], self.cameras, self.gray)
        # --> list of (list of images that form each of the reference lightfield images)
        ref_lfs = [load_lightfield(ref_img, self.cameras, self.gray) for ref_img in sample['ref_imgs']]

        tgt_stack = None
        ref_stacks = None
        if self.cameras_stacked == 'full':
            # load target and reference lightfields from all the cameras as a stack
            tgt_stack = load_stacked_epi_no_repeats(sample['tgt'], same_parallax=False)
            ref_stacks = [load_stacked_epi_no_repeats(ref_img, same_parallax=False) for ref_img in sample['ref_imgs']]
        elif self.cameras_stacked == 'input':
            tgt_stack = load_lightfield(sample['tgt'], self.cameras_stacked_indices, self.gray)
            ref_stacks = [load_lightfield(ref_img, self.cameras_stacked_indices, self.gray)
                          for ref_img in sample['ref_imgs']]

        # load ground truth pose of target frame wrt reference frames - [refs x 4x4 matrix]
        poses_gt_refs_tgt = torch.Tensor([load_relative_pose(ref, sample['tgt'], self.no_pose)
                                          for ref in sample['ref_imgs']])
        # intrinsics = np.copy(sample['intrinsics'])

        # apply transforms if present
        if self.transform is not None:
            tgt_lf, _ = self.transform(tgt_lf, np.zeros((3, 3)))
            ref_lfs = [self.transform(ref, np.zeros((3, 3)))[0] for ref in ref_lfs]

            tgt_stack, _ = self.transform(tgt_stack, np.zeros((3, 3)))
            ref_stacks = [self.transform(ref, np.zeros((3, 3)))[0] for ref in ref_stacks]

        # concatenate light fields on colour channel
        tgt_lf = torch.cat(tuple(tgt_lf), 0)
        ref_lfs = [torch.cat(ref, 0) for ref in ref_lfs]

        tgt_stack = torch.cat(tuple(tgt_stack), 0)
        ref_stacks = [torch.cat(ref, 0) for ref in ref_stacks]

        # Create metadata and training data
        metadata = MetaData(self.cameras, sample['tgt'], sample['ref_imgs'], self.gray, False)

        trainingdata = TrainingData(tgt_lf, tgt_stack, ref_lfs, ref_stacks, sample['intrinsics'],
                                    poses_gt_refs_tgt, metadata)
        # if self.cameras_stacked == 'full':
        #
        #     trainingdata = TrainingData(tgt_lf, tgt_stack, ref_lfs, ref_stacks, sample['intrinsics'], pose, metadata)
        # else:
        #     # the stacks sent as input for the pose and disp nets are the same as the stack used for photometric error
        #     trainingdata = TrainingData(tgt_lf, tgt_lf, ref_lfs, ref_lfs, sample['intrinsics'], pose, metadata)

        return trainingdata.get_as_dict()


class TiledEPILoader(BaseDataset):
    """
    Class for loading tiles Epipolar Plane Images (EPI)
    """

    def __init__(self, root, cameras, gray=False, seed=None, train=True, sequence_length=3,
                 transform=None, shuffle=True, sequence=None, cameras_epi='vertical', no_pose=False):
        """
        Initlializer

        :param root: Path to the folder where the data is present.
        This folder should have subfolders seq#, train.txt, val.txt
        :type root: str
        :param cameras: List of camera indices to use for computing the multi-warp photometric error.
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
        :type transform: tensor
        :param shuffle: Boolean indicating if the data has to be shuffled. Default is True for training.
        :type shuffle: bool
        :param sequence: Sequence index that has to be loaded, if required. Default is None to load all data.
        :type sequence: int
        :param cameras_epi: String indicating what cameras to use in forming the EPI. 'vertical', 'horizontal', 'full'
        :type cameras_epi: str
        """
        
        super(TiledEPILoader, self).__init__(root, cameras, gray, seed, train, sequence_length, 
                                             transform, shuffle, sequence)
        assert cameras_epi in ["vertical", "horizontal", "full"]
        self.cameras_epi = cameras_epi
        self.cameras_stacked_indices = [3, 8, 13, 7, 9]
        self.no_pose = no_pose

    def __getitem__(self, index):
        """
        Returns the element at the desired index from the dataset.

        :param index: Index of the element of the dataset
        :type index: int
        :return: Dictionary of training data
        :rtype: dict
        """

        # get the data sample at the specified index
        sample = self.samples[index]
        # load target and reference light fields for photometric error computation
        tgt_lf = load_lightfield(sample['tgt'], self.cameras, self.gray)
        ref_lfs = [load_lightfield(ref_img, self.cameras, self.gray) for ref_img in sample['ref_imgs']]
        # load target and reference light fields for sending to encoders
        tgt_stack = load_lightfield(sample['tgt'], self.cameras_stacked_indices, self.gray)
        ref_stacks = [load_lightfield(ref_img, self.cameras_stacked_indices, self.gray)
                      for ref_img in sample['ref_imgs']]

        # load epi images
        tgt_epi_h = tgt_epi_v = tgt_epi = None
        ref_epis_h = ref_epis_v = ref_epis = None

        if self.cameras_epi == 'vertical':
            # the full epi is directly only the vertical epi
            tgt_epi = load_tiled_epi_vertical(sample['tgt'])
            ref_epis = [load_tiled_epi_vertical(ref_img) for ref_img in sample['ref_imgs']]
        elif self.cameras_epi == 'horizontal':
            # the full epi is directly only the horizontal epi
            tgt_epi = load_tiled_epi_horizontal(sample['tgt'])
            ref_epis = [load_tiled_epi_horizontal(ref_img) for ref_img in sample['ref_imgs']]
        elif self.cameras_epi == 'full':
            # horizontal and vertical epis need to be separately handled
            tgt_epi_v, tgt_epi_h = load_tiled_epi_full(sample['tgt'])
            # this will be a list of tuples
            ref_epis_v = []
            ref_epis_h = []
            for ref_img in sample['ref_imgs']:
                ref_epi_v, ref_epi_h = load_tiled_epi_full(ref_img)
                ref_epis_v.append(ref_epi_v)
                ref_epis_h.append(ref_epi_h)

        # load ground truth pose of each reference frame wrt target frame - [refs x 4x4 matrix]
        poses_gt_refs_tgt = torch.Tensor([load_relative_pose(ref, sample['tgt'], self.no_pose)
                                          for ref in sample['ref_imgs']])
        # intrinsics = np.copy(sample['intrinsics'])

        # apply transforms if present
        if self.transform is not None:
            # apply transforms on the images photometric error
            tgt_lf, _ = self.transform(tgt_lf, np.zeros((3, 3)))
            ref_lfs = [self.transform(ref, np.zeros((3, 3)))[0] for ref in ref_lfs]

            tgt_stack, _ = self.transform(tgt_stack, np.zeros((3, 3)))
            ref_stacks = [self.transform(ref, np.zeros((3, 3)))[0] for ref in ref_stacks]

            # apply transforms on the encoded light fields
            if self.cameras_epi == 'full':
                # apply separately on the horizontal and vertical images
                tgt_epi_h, _ = self.transform(tgt_epi_h, np.zeros((3, 3)))
                tgt_epi_v, _ = self.transform(tgt_epi_v, np.zeros((3, 3)))

                ref_epis_h = [self.transform(ref, np.zeros((3, 3)))[0] for ref in ref_epis_h]
                ref_epis_v = [self.transform(ref, np.zeros((3, 3)))[0] for ref in ref_epis_v]
            else:
                tgt_epi, _ = self.transform(tgt_epi, np.zeros((3, 3)))
                ref_epis = [self.transform(ref, np.zeros((3, 3)))[0] for ref in ref_epis]

        # concatenate light fields on colour channel
        tgt_lf = torch.cat(tuple(tgt_lf), 0)
        ref_lfs = [torch.cat(ref, 0) for ref in ref_lfs]

        tgt_stack = torch.cat(tuple(tgt_stack), 0)
        ref_stacks = [torch.cat(ref, 0) for ref in ref_stacks]

        if self.cameras_epi == 'full':
            tgt_epi_h = torch.cat(tuple(tgt_epi_h), 0)
            tgt_epi_v = torch.cat(tuple(tgt_epi_v), 0)

            ref_epis_h = [torch.cat(ref, 0) for ref in ref_epis_h]
            ref_epis_v = [torch.cat(ref, 0) for ref in ref_epis_v]
        else:
            tgt_epi = torch.cat(tuple(tgt_epi), 0)
            ref_epis = [torch.cat(ref, 0) for ref in ref_epis]

        # Create metadata and training data
        metadata = MetaData(self.cameras, sample['tgt'], sample['ref_imgs'], self.gray, False)
        if self.cameras_epi == 'full':
            trainingdata = TrainingDataEpi(tgt=tgt_lf, tgt_formatted_h=tgt_epi_h, tgt_formatted_v=tgt_epi_v,
                                           ref=ref_lfs, ref_formatted_h=ref_epis_h, ref_formatted_v=ref_epis_v,
                                           tgt_stack=tgt_stack, ref_stacks=ref_stacks,
                                           intrinsics=sample['intrinsics'],
                                           poses_gt_refs_tgt=poses_gt_refs_tgt, metadata=metadata)
        else:
            trainingdata = TrainingData(tgt=tgt_lf, tgt_formatted=tgt_epi, ref=ref_lfs, ref_formatted=ref_epis,
                                        intrinsics=sample['intrinsics'], poses_gt_refs_tgt=poses_gt_refs_tgt,
                                        metadata=metadata)

        return trainingdata.get_as_dict()


def get_epi_loaders(args, train_transform, valid_transform, shuffle=True, no_pose=False):
    """
    Returns the dataset objects with epipolar plane images

    :param args: user specified arguments
    :type args: argparse object
    :param train_transform: Transform to apply on the training images
    :type train_transform: Tensor
    :param valid_transform: Transform to apply on the validation images
    :type valid_transform: Tensor
    :param shuffle: Boolean indicating if data has to be shuffled
    :type shuffle: bool
    :param no_pose: Boolean indicating if pose is present or not. Default False
    :type no_pose: bool
    :return: Training and Validation datasets
    :rtype: tuple of objects
    """
    train_set = TiledEPILoader(
        args.data,
        cameras=args.cameras,
        gray=True,      # TODO: Why is EPI loader always grayscale? Is it to reduce computation?
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length,
        transform=train_transform,
        shuffle=shuffle,
        cameras_epi=args.cameras_epi,
        no_pose=no_pose
    )

    val_set = TiledEPILoader(
        args.data,
        cameras=args.cameras,
        gray=True,      # TODO: Why is EPI loader always grayscale? Is it to reduce computation?
        seed=args.seed,
        train=False,
        sequence_length=args.sequence_length,
        transform=valid_transform,
        shuffle=False,       # shuffle is false for validation
        cameras_epi=args.cameras_epi,
        no_pose=no_pose
    )

    return train_set, val_set


def get_validation_epi_loader(args, sequence=None, transform=None, no_pose=False):
    """
    Returns the dataset objects with epipolar plane images for validation

    :param args:user specified arguments
    :type args: argparse object
    :param sequence: Index of the sequence to use for validation
    :type sequence: int
    :param transform: Transform to apply on the images
    :type transform: Tensor
    :param no_pose: Boolean indicating if pose is present or not. Default False
    :type no_pose: bool
    :return: Validation dataset
    :rtype: object
    """
    return TiledEPILoader(
        args.data,
        cameras=args.cameras,
        gray=True,
        seed=args.seed,
        train=False,
        sequence_length=args.sequence_length,
        transform=transform,
        shuffle=False,       # shuffle is false for validation
        sequence=sequence,
        cameras_epi=args.cameras_epi,
        no_pose=no_pose
    )


def get_focal_stack_loaders(args, train_transform, valid_transform, shuffle=True, no_pose=False):
    """
    Returns the dataset objects with light field focal stacks.

    :param args: user specified arguments
    :type args: argparse object
    :param train_transform: Transform to apply on the training images
    :type train_transform: Tensor
    :param valid_transform: Transform to apply on the validation images
    :type valid_transform: Tensor
    :param shuffle: Boolean indicating if data has to be shuffled
    :type shuffle: bool
    :param no_pose: Boolean indicating if pose is present or not. Default False
    :type no_pose: bool
    :return: Training and Validation datasets
    :rtype: tuple of objects
    """
    train_set = FocalstackLoader(
        args.data,
        cameras=args.cameras,
        gray=args.gray,
        transform=train_transform,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length,
        fs_num_cameras=args.num_cameras,
        fs_num_planes=args.num_planes,
        shuffle=shuffle,
        no_pose=no_pose
    )

    val_set = FocalstackLoader(
        args.data,
        cameras=args.cameras,
        gray=args.gray,
        transform=valid_transform,
        seed=args.seed,
        train=False,
        sequence_length=args.sequence_length,
        fs_num_cameras=args.num_cameras,
        fs_num_planes=args.num_planes,
        shuffle=False,       # shuffle is false for validation
        no_pose=no_pose
    )

    return train_set, val_set


def get_validation_focal_stack_loader(args, sequence=None, transform=None, no_pose=False):
    """
    Returns the dataset objects with images as light field focal stacks for validation

    :param args:user specified arguments
    :type args: argparse object
    :param sequence: Index of the sequence to use for validation
    :type sequence: int
    :param transform: Transform to apply on the images
    :type transform: Tensor
    :param no_pose: Boolean indicating if pose is present or not. Default False
    :type no_pose: bool
    :return: Validation dataset
    :rtype: object
    """
    val_set = FocalstackLoader(
        args.data,
        cameras=args.cameras,
        gray=args.gray,
        transform=transform,
        seed=args.seed,
        train=False,
        sequence_length=args.sequence_length,
        fs_num_cameras=args.num_cameras,
        fs_num_planes=args.num_planes,
        shuffle=False,       # shuffle is false for validation
        sequence=sequence,
        no_pose=no_pose
    )

    return val_set


def get_stacked_lf_loaders(args, train_transform, valid_transform, shuffle=True, no_pose=False, is_monocular=False):
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
    :param no_pose: Boolean indicating if pose is present or not. Default False
    :type no_pose: bool
    :param is_monocular: Boolean indicating if it is monocular mode of operation. Default False
    :type is_monocular: bool
    :return: Training and Validation datasets
    :rtype: tuple of objects
    """

    train_set = StackedLFLoader(
        args.data,
        cameras=args.cameras,
        gray=args.gray,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length,
        transform=train_transform,
        shuffle=shuffle,
        cameras_stacked=args.cameras_stacked,
        no_pose=no_pose,
        is_monocular=is_monocular
    )

    val_set = StackedLFLoader(
        args.data,
        cameras=args.cameras,
        gray=args.gray,
        seed=args.seed,
        train=False,
        sequence_length=args.sequence_length,
        transform=valid_transform,
        shuffle=False,       # shuffle is false for validation
        cameras_stacked=args.cameras_stacked,
        no_pose=no_pose,
        is_monocular=is_monocular
    )

    return train_set, val_set


def get_validation_stacked_lf_loader(args, sequence=None, transform=None, no_pose=False, is_monocular=False):
    """
    Returns the dataset objects with images as light field image stacks for validation

    :param args:user specified arguments
    :type args: argparse object
    :param sequence: Index of the sequence to use for validation
    :type sequence: int
    :param transform: Transform to apply on the images
    :type transform: Tensor
    :param no_pose: Boolean indicating if pose is present or not. Default False
    :type no_pose: bool
    :param is_monocular: Boolean indicating if it is monocular mode of operation. Default False
    :type is_monocular: bool
    :return: Validation dataset
    :rtype: object
    """
    return StackedLFLoader(
        args.data,
        cameras=args.cameras,
        gray=args.gray,
        seed=args.seed,
        train=False,
        sequence_length=args.sequence_length,
        transform=transform,
        shuffle=False,       # shuffle is false for validation
        sequence=sequence,
        cameras_stacked=args.cameras_stacked,
        no_pose=no_pose,
        is_monocular=is_monocular
    )
