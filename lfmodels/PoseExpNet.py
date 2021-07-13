import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_


class LFPoseNet(nn.Module):
    """
    PoseNet for LightField images
    """

    def __init__(self, in_channels=3, nb_ref_imgs=2, encoder=None):
        """
        Initializer. Defines the layers of the network model.

        :param in_channels: Number of channels in the input image
        :type in_channels:  int
        :param nb_ref_imgs: number of reference images to use
        :type nb_ref_imgs: int
        :param encoder: Encoder for lightfield images - used for encoding images in the EPI format
        :type encoder: RelativeEpiEncoder
        """
        super(LFPoseNet, self).__init__()
        self.nb_ref_imgs = nb_ref_imgs
        self.encoder = encoder

        def conv_pn(in_planes, out_planes, kernel_size=3):
            """
            Convolutional Layer followed by ReLU

            :param in_planes: number of channels in the input
            :type in_planes: int
            :param out_planes: number of channels in the output
            :type out_planes: int
            :param kernel_size: Size of the convolving kernel
            :type kernel_size: int or tuple
            :return: the output of the layer
            :rtype: tensor
            """
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=kernel_size,
                          padding=(kernel_size - 1) // 2,
                          stride=2),
                nn.ReLU(inplace=True)
            )

        # define the convolutional + ReLU layers [default kernel_size=3, stride=2, padding = (kernel_size -1)//2]
        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv_pn(in_channels*(1+self.nb_ref_imgs), conv_planes[0], kernel_size=7)
        self.conv2 = conv_pn(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv_pn(conv_planes[1], conv_planes[2])
        self.conv4 = conv_pn(conv_planes[2], conv_planes[3])
        self.conv5 = conv_pn(conv_planes[3], conv_planes[4])
        self.conv6 = conv_pn(conv_planes[4], conv_planes[5])
        self.conv7 = conv_pn(conv_planes[5], conv_planes[6])

        # final layer is a 1x1 convolutional layer
        self.pose_pred = nn.Conv2d(conv_planes[6], 6*self.nb_ref_imgs, kernel_size=1, padding=0)

    def init_weights(self):
        """
        Initializes weights with Xavier Uniform weights
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)

    def encode(self, tgt_formatted, tgt_unformatted, ref_formatted, ref_unformatted,
               tgt_formatted_ex=None, ref_formatted_ex=None):
        """
        Encodes the lightfield images

        :param tgt_formatted: the target tiled epipolar image [B, 1, H, W*tilesize] or [B, 1, H*tilesize, W]
        :type tgt_formatted: tensor
        :param tgt_unformatted: the target grid of images stacked on the colour-channel   [B, N, H, W]
        :type tgt_unformatted: tensor
        :param ref_formatted: list of reference tiled epipolar image [B, 1, H, W*tilesize] or [B, 1, H*tilesize, W]
        :type ref_formatted: tensor
        :param ref_unformatted: list of grid of images stacked on the colour-channel   [B, N, H, W]
        :type ref_unformatted: tensor
        :return: the encoded target image concatenated with the stacked image-grid   [B, N+16, H, W],
         the same for each of the images of the list of reference images
        :rtype: tuple of tensor, list of tensors
        """
        return self.encoder(tgt_formatted, tgt_unformatted, ref_formatted, ref_unformatted,
                            tgt_formatted_ex, ref_formatted_ex)
    
    def has_encoder(self):
        """
        Returns True if encoder is not None

        :return: True if encoder is not None else False
        :rtype: bool
        """
        return False if self.encoder is None else True

    def forward(self, target_image, ref_imgs, rev=False):
        """
        Forward pass of the network

        :param target_image: Target image   [B, channels, h, w]
        :type target_image: tensor
        :param ref_imgs: List of reference images   list of images of shape [B, channels, h, w]
        :type ref_imgs: list of tensors
        :return: the 6DOF pose of the target frame relative to the reference frames
        :rtype: tensor
        """
        if rev:
            assert (len(target_image) == self.nb_ref_imgs)
            ref_imgs = [ref_imgs]
            i_input = target_image
        else:
            assert(len(ref_imgs) == self.nb_ref_imgs)
            i_input = [target_image]    # make a list of encoded target image
        i_input.extend(ref_imgs)    # extend the list with encoded reference images
        i_input = torch.cat(i_input, 1)     # convert it to a tensor
        out_conv1 = self.conv1(i_input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        pose = self.pose_pred(out_conv7)
        pose = pose.mean(3).mean(2)
        pose = 0.01 * pose.view(pose.size(0), self.nb_ref_imgs, 6)

        return pose
