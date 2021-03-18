import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_


class RelativeEpiEncoder(nn.Module):
    """
    Learns an encoding for epipolar plane images to use in pose estimation
    """

    def __init__(self, direction, tilesize):
        """
        Initializer. Defines the layers that define the network model.

        :param direction: String indicating the direction in which slices were taken. "vertical" or "horizontal"
        :type direction: str
        :param tilesize: width of the image for vertical tiling, height of the image for horizontal tiling.
        :type tilesize: int
        :return: neural layer that encodes an epi polar plane image (EPI)
        :rtype: pytorch layer
        """
        assert direction in ['vertical', 'horizontal', 'full']
        super(RelativeEpiEncoder, self).__init__()
        self.direction = direction

        self.conv1 = None
        self.conv1_ex = None
        if direction == 'vertical':
            self.conv1 = nn.Conv2d(1, 16, kernel_size=tilesize,
                                   stride=(1, tilesize),    # horizontal stride is tilesize
                                   padding=(tilesize // 2, 0))
        elif direction == 'horizontal':
            self.conv1 = nn.Conv2d(1, 16, kernel_size=tilesize,
                                   stride=(tilesize, 1),    # vertical stride is tilesize
                                   padding=(0, tilesize // 2))
        elif direction == 'full':
            # conv to apply on epi from vertical cameras
            self.conv1 = nn.Conv2d(1, 16, kernel_size=tilesize,
                                   stride=(1, tilesize),  # horizontal stride is tilesize
                                   padding=(tilesize // 2, 0))
            # conv to apply on epi from horizontal cameras
            self.conv1_ex = nn.Conv2d(1, 16, kernel_size=tilesize,
                                      stride=(tilesize, 1),    # vertical stride is tilesize
                                      padding=(0, tilesize // 2))

        self.relu1 = nn.ReLU(inplace=True)

    def encode(self, formatted, stacked, formatted_ex=None):
        """
        Function that performs the encoding operation using a CNN

        :param formatted: if direction is "vertical" or "full" the tiled EPI from vertical cameras. A wide tensor
            [B, 1, H, W*tilesize]. if direction is "horizontal" the tile EPI from horizontal cameras. A tall tensor
            [B, 1, H*tilesize, W]
        :type formatted: tensor
        :param stacked: list of lightfield sub-apertures stacked on the colour-channel   [B, N, H, W]
        :type stacked: tensor
        :return: the encoded light field concatenated with the stacked image-grid   [B, N+16, H, W]
        :rtype: tensor
        :param formatted_ex: Only if direction is "full" the tiled EPI from horizontal cameras. A tall tensor
            [B, 1, H*tilesize, W]. None by default.
        :type formatted_ex: tensor
        """

        if formatted_ex is None:
            inp_height, inp_width = formatted.shape[2:]
            x = self.conv1(formatted)
            x = self.relu1(x)

            if self.direction == 'vertical':
                x = x[:, :, 0:inp_height, :]
            elif self.direction == 'horizontal':
                x = x[:, :, :, 0:inp_width]

            if stacked is not None:
                x = torch.cat([x, stacked], dim=1)
            return x
        if formatted_ex is not None:
            inp_height_v, inp_width_v = formatted.shape[2:]
            inp_height_h, inp_width_h = formatted_ex.shape[2:]
            x_v = self.conv1(formatted)
            x_v = self.relu1(x_v)
            x_h = self.conv1_ex(formatted_ex)
            x_h = self.relu1(x_h)

            x_v = x_v[:, :, 0:inp_height_v, :]
            x_h = x_h[:, :, :, 0:inp_width_h]

            if stacked is not None:
                x = torch.cat([x_v, x_h, stacked], dim=1)
            else:
                x = torch.cat([x_v, x_h], dim=1)
            return x

    def forward(self, tgt_lf_formatted, tgt_lf_stacked, ref_lfs_formatted, ref_lfs_stacked,
                tgt_lf_formatted_ex=None, ref_lfs_formatted_ex=None):
        """
        Defines the forward pass

        :param tgt_lf_formatted: If direction is "vertical" or "full" the target tiled EPI from vertical cameras.
            It is then a wide tensor [B, 1, H, W*tilesize]. If direction is "horizontal" the tile EPI from horizontal
            cameras. Then it is tall tensor [B, 1, H*tilesize, W].
        :type tgt_lf_formatted: tensor
        :param tgt_lf_stacked: the target lightfield sub-aperture images stacked on the colour-channel   [B, N, H, W]
        :type tgt_lf_stacked: tensor
        :param ref_lfs_formatted: If direction is "vertical" or "full" the list of reference tiled EPI from vertical
            cameras. It is then a wide tensor [B, 1, H, W*tilesize]. If direction is "horizontal" the tile EPI from
            horizontal cameras. Then it is tall tensor [B, 1, H*tilesize, W].
        :type ref_lfs_formatted: list
        :param ref_lfs_stacked: list of lightfield sub-apertures stacked on the colour-channel   [B, N, H, W]
        :type ref_lfs_stacked: list
        :param tgt_lf_formatted_ex: If direction is "full" the target tiled EPI from the horizontal cameras. Then it is
            a tall tensor [B, 1, H*tilesize, W].
        :type tgt_lf_formatted_ex: tensor
        :param ref_lfs_formatted_ex: If direction is "full" the list of reference tiled EPI from the horizontal cameras.
            Then it is a tall tensor [B, 1, H*tilesize, W].
        :type ref_lfs_formatted_ex: list
        :return: the encoded target image - light field images and stack concatenated   [B, N+16, H, W],
            the same for each of the images of the list of reference images
        :rtype: tuple of tensor, list of tensors
        """

        if tgt_lf_formatted_ex is None and ref_lfs_formatted_ex is None:
            tgt = self.encode(tgt_lf_formatted, tgt_lf_stacked, None)
            ref = [self.encode(formatted=formatted, stacked=stacked)
                   for formatted, stacked in zip(ref_lfs_formatted, ref_lfs_stacked)]
            return tgt, ref
        else:
            # both should not be None
            assert tgt_lf_formatted_ex is not None and ref_lfs_formatted_ex is not None
            tgt = self.encode(tgt_lf_formatted, tgt_lf_stacked, tgt_lf_formatted_ex)
            ref = [self.encode(formatted=formatted,
                               stacked=stacked,
                               formatted_ex=formatted_ex) for formatted, stacked, formatted_ex
                   in zip(ref_lfs_formatted, ref_lfs_stacked, ref_lfs_formatted_ex)]
            return tgt, ref


class EpiEncoder(nn.Module):
    """
    Learns an encoding of epipolar images when presented as a 2D grid of epipolar slices.
    This is the encoding used in the disparity estimation.
    """
 
    def __init__(self, direction, tilesize):
        """
        Initializer

        :param direction: String indicating the direction in which slices were taken. "vertical" or "horizontal"
        :type direction: str
        :param tilesize: width of the image for vertical tiling, height of the image for horizontal tiling.
        :type tilesize: int
        :return: neural layer that encodes an epi polar plane image (EPI)
        :rtype: pytorch layer
        """
        super(EpiEncoder, self).__init__()
        assert direction in ['vertical', 'horizontal', 'full']
        self.direction = direction

        self.conv1 = None
        self.conv1_ex = None
        if direction == 'vertical':
            self.conv1 = nn.Conv2d(1, 16, kernel_size=tilesize,
                                   stride=(1, tilesize),    # horizontal stride is tilesize
                                   padding=(tilesize // 2, 0))
        elif direction == 'horizontal':
            self.conv1 = nn.Conv2d(1, 16, kernel_size=tilesize,
                                   stride=(tilesize, 1),    # vertical stride is tilesize
                                   padding=(0, tilesize // 2))
        elif direction == 'full':
            # conv to apply on epi from vertical cameras
            self.conv1 = nn.Conv2d(1, 16, kernel_size=tilesize,
                                   stride=(1, tilesize),  # horizontal stride is tilesize
                                   padding=(tilesize // 2, 0))
            # conv to apply on epi from horizontal cameras
            self.conv1_ex = nn.Conv2d(1, 16, kernel_size=tilesize,
                                      stride=(tilesize, 1),    # vertical stride is tilesize
                                      padding=(0, tilesize // 2))
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, lf_formatted, lf_stacked, lf_formatted_ex=None):
        """
        Defines the forward pass of the layer when the direction chosen is "full".

        :param lf_formatted: if direction was "vertical" or "full", the tiled EPI for vertical cameras. A wide tensor
            [B, 1, H, W*tilesize]. If the direction was "horizontal", the tiled EPI for the horizontal cameras. A tall
            tensor [B, 1, H*tilesize, W]
        :type lf_formatted: tensor
        :param lf_formatted_ex: only if direction was "full", the tiled EPI for horizontal cameras. A tall tensor
            [B, 1, H*tilesize, W]. None by default.
        :type lf_formatted_ex: tensor
        :param lf_stacked: the grid of images stacked on the colour-channel   [B, N, H, W]
        :type lf_stacked: tensor
        :return: the encoded light field concatenated with the stacked image-grid   [B, N+16, H, W]
        :rtype: tensor
        """

        if lf_formatted_ex is None:
            assert lf_formatted is not None
            inp_height, inp_width = lf_formatted.shape[2:]
            # apply appropriate convolution followed by relu
            x = self.conv1(lf_formatted)
            x = self.relu1(x)

            # crop to appropriate shape
            if self.direction == 'vertical':
                x = x[:, :, 0:inp_height, :]
            if self.direction == 'horizontal':
                x = x[:, :, :, 0:inp_width]

            if lf_stacked is not None:
                # concatenate the stack
                x = torch.cat([x, lf_stacked], dim=1)
            return x
        else:
            assert lf_formatted is not None and lf_formatted_ex is not None
            inp_height_v, inp_width_v = lf_formatted.shape[2:]
            inp_height_h, inp_width_h = lf_formatted_ex.shape[2:]
            # apply appropriate convolution followed by relu
            x_v = self.conv1(lf_formatted)
            x_v = self.relu1(x_v)
            x_h = self.conv1_ex(lf_formatted_ex)
            x_h = self.relu1(x_h)
            # crop to the appropriate shape
            x_v = x_v[:, :, 0:inp_height_v, :]
            x_h = x_h[:, :, :, 0:inp_width_h]

            if lf_stacked is not None:
                # concatenate the two convolved outputs and the stack
                x = torch.cat([x_v, x_h, lf_stacked], dim=1)
            else:
                # concatenate only the two convolved outputs
                x = torch.cat([x_v, x_h], dim=1)
            return x


""" Demo showing operation of encoder """
if __name__ == "__main__":

    num_cameras = 8
    height = 200
    width = 200
    batch_size = 4

    stack = torch.rand([batch_size, num_cameras, height, width])

    net_v = EpiEncoder('vertical', num_cameras)
    vertical_tiled_epi = torch.rand([batch_size, 1, height, width * num_cameras])
    out_v = net_v(vertical_tiled_epi, stack)
    print(out_v.shape)

    horizontal_tiled_epi = torch.rand([batch_size, 1, height * num_cameras, width])
    net_h = EpiEncoder('horizontal', num_cameras)
    out_h = net_h(horizontal_tiled_epi, stack)
    print(out_h.shape)

    net = EpiEncoder('full', num_cameras)
    out = net(vertical_tiled_epi, stack, horizontal_tiled_epi)
    print(out.shape)
