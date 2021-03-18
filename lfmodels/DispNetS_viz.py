import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_


class LFDispNetViz(nn.Module):
    """
    DispNet for LightField images
    """

    def __init__(self, in_channels=3, out_channels=1, alpha=10, beta=0.01, encoder=None):
        """
        Initializer. Defines the layers of the network model.

        :param in_channels: Number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param alpha: Some scale factor
        :type alpha: float
        :param beta: Some other scale factor
        :type beta: float
        :param encoder: Encoder for lightfield images - used for encoding images in the EPI format
        :type encoder: EpiEncoder
        """
        super(LFDispNetViz, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.encoder = encoder

        def downsample_conv(in_planes, out_planes, kernel_size=3):
            """
            Downsampling convolution

            :param in_planes: number of input channels
            :type in_planes: int
            :param out_planes: number of output channels
            :type out_planes: int
            :param kernel_size: size of the convolving kernel
            :type kernel_size: int or tuple
            :return: the output of the layer
            :rtype: tensor
            """
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size - 1) // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
                nn.ReLU(inplace=True)
            )

        def predict_disp(in_planes, out_planes):
            """
            Predict disparity - convolution followed by a sigmoid function

            :param in_planes: number of input channels
            :type in_planes: int
            :param out_planes: number of output channels
            :type out_planes: int
            :return: the output of the layer
            :rtype: tensor
            """
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
                nn.Sigmoid()
            )

        def conv_dn(in_planes, out_planes):
            """
            Convolutional Layer followed by ReLU - kernel_size=3, padding=1

            :param in_planes: number of channels in the input
            :type in_planes: int
            :param out_planes: number of channels in the output
            :type out_planes: int
            :return: the output of the layer
            :rtype: tensor
            """
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        def upconv_dn(in_planes, out_planes):
            """
            Upconvolution
            :param in_planes: number of channels in the input
            :type in_planes: int
            :param out_planes: number of channels in the output
            :type out_planes: int
            :return: the output of the layer
            :rtype: tensor
            """
            return nn.Sequential(
                nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3,
                                   stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True)
            )

        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        self.conv1 = downsample_conv(in_channels,    conv_planes[0], kernel_size=7)
        self.conv2 = downsample_conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = downsample_conv(conv_planes[1], conv_planes[2])
        self.conv4 = downsample_conv(conv_planes[2], conv_planes[3])
        self.conv5 = downsample_conv(conv_planes[3], conv_planes[4])
        self.conv6 = downsample_conv(conv_planes[4], conv_planes[5])
        self.conv7 = downsample_conv(conv_planes[5], conv_planes[6])

        upconv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.upconv7 = upconv_dn(conv_planes[6],   upconv_planes[0])
        self.upconv6 = upconv_dn(upconv_planes[0], upconv_planes[1])
        self.upconv5 = upconv_dn(upconv_planes[1], upconv_planes[2])
        self.upconv4 = upconv_dn(upconv_planes[2], upconv_planes[3])
        self.upconv3 = upconv_dn(upconv_planes[3], upconv_planes[4])
        self.upconv2 = upconv_dn(upconv_planes[4], upconv_planes[5])
        self.upconv1 = upconv_dn(upconv_planes[5], upconv_planes[6])

        self.iconv7 = conv_dn(upconv_planes[0] + conv_planes[5], upconv_planes[0])
        self.iconv6 = conv_dn(upconv_planes[1] + conv_planes[4], upconv_planes[1])
        self.iconv5 = conv_dn(upconv_planes[2] + conv_planes[3], upconv_planes[2])
        self.iconv4 = conv_dn(upconv_planes[3] + conv_planes[2], upconv_planes[3])
        self.iconv3 = conv_dn(out_channels + upconv_planes[4] + conv_planes[1], upconv_planes[4])
        self.iconv2 = conv_dn(out_channels + upconv_planes[5] + conv_planes[0], upconv_planes[5])
        self.iconv1 = conv_dn(out_channels + upconv_planes[6], upconv_planes[6])

        self.predict_disp4 = predict_disp(upconv_planes[3], out_channels)
        self.predict_disp3 = predict_disp(upconv_planes[4], out_channels)
        self.predict_disp2 = predict_disp(upconv_planes[5], out_channels)
        self.predict_disp1 = predict_disp(upconv_planes[6], out_channels)

    def init_weights(self):
        """
        Initializes weights with Xavier Uniform weights
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def encode(self, formatted, unformatted, formatted_ex=None):
        """
        Encodes the lightfield images

        :param formatted: tiled epipolar image [B, 1, H, W*tilesize] or [B, 1, H*tilesize, W]
        :type formatted: tensor
        :param unformatted: grid of images stacked on the colour-channel   [B, N, H, W]
        :type unformatted: tensor
        :return: the encoded target image concatenated with the stacked image-grid   [B, N+16, H, W]
        :rtype: tensor
        """
        return self.encoder(formatted, unformatted, formatted_ex)

    def has_encoder(self):
        """
        Returns True if encoder is not None

        :return: True if encoder is not None else False
        :rtype: bool
        """
        return False if self.encoder is None else True

    @staticmethod
    def crop_top_left(input_tensor, ref):
        """
        Takes a crop of the size of the reference image from the the input tensor.
        The crop is taken from the top left corner of the input tensor, not central crop!

        :param input_tensor: input tensor
        :type input_tensor: tensor
        :param ref: reference image
        :type ref: tensor
        :return: cropped tensor
        :rtype: tensor
        """
        assert (input_tensor.size(2) >= ref.size(2) and input_tensor.size(3) >= ref.size(3))
        return input_tensor[:, :, :ref.size(2), :ref.size(3)]

    def forward(self, x):
        """
        Forward pass of the network

        :param x: input tensor
        :type x: tensor
        :return: disparity image (or images)
        :rtype: tensor
        """
        self.out_conv1 = self.conv1(x)
        self.out_conv2 = self.conv2(self.out_conv1)
        self.out_conv3 = self.conv3(self.out_conv2)
        self.out_conv4 = self.conv4(self.out_conv3)
        self.out_conv5 = self.conv5(self.out_conv4)
        self.out_conv6 = self.conv6(self.out_conv5)
        self.out_conv7 = self.conv7(self.out_conv6)

        self.out_upconv7 = self.crop_top_left(self.upconv7(self.out_conv7), self.out_conv6)
        self.concat7 = torch.cat((self.out_upconv7, self.out_conv6), 1)
        self.out_iconv7 = self.iconv7(self.concat7)

        self.out_upconv6 = self.crop_top_left(self.upconv6(self.out_iconv7), self.out_conv5)
        self.concat6 = torch.cat((self.out_upconv6, self.out_conv5), 1)
        self.out_iconv6 = self.iconv6(self.concat6)

        self.out_upconv5 = self.crop_top_left(self.upconv5(self.out_iconv6), self.out_conv4)
        self.concat5 = torch.cat((self.out_upconv5, self.out_conv4), 1)
        self.out_iconv5 = self.iconv5(self.concat5)

        self.out_upconv4 = self.crop_top_left(self.upconv4(self.out_iconv5), self.out_conv3)
        self.concat4 = torch.cat((self.out_upconv4, self.out_conv3), 1)
        self.out_iconv4 = self.iconv4(self.concat4)
        self.disp4 = self.alpha * self.predict_disp4(self.out_iconv4) + self.beta

        self.out_upconv3 = self.crop_top_left(self.upconv3(self.out_iconv4), self.out_conv2)
        self.disp4_up = self.crop_top_left(torch.nn.functional.interpolate(self.disp4,
                                                                      scale_factor=2,
                                                                      mode='bilinear',
                                                                      align_corners=False), self.out_conv2)
        self.concat3 = torch.cat((self.out_upconv3, self.out_conv2, self.disp4_up), 1)
        self.out_iconv3 = self.iconv3(self.concat3)
        self.disp3 = self.alpha * self.predict_disp3(self.out_iconv3) + self.beta

        self.out_upconv2 = self.crop_top_left(self.upconv2(self.out_iconv3), self.out_conv1)
        self.disp3_up = self.crop_top_left(torch.nn.functional.interpolate(self.disp3,
                                                                      scale_factor=2,
                                                                      mode='bilinear',
                                                                      align_corners=False), self.out_conv1)
        self.concat2 = torch.cat((self.out_upconv2, self.out_conv1, self.disp3_up), 1)
        self.out_iconv2 = self.iconv2(self.concat2)
        self.disp2 = self.alpha * self.predict_disp2(self.out_iconv2) + self.beta

        self.out_upconv1 = self.crop_top_left(self.upconv1(self.out_iconv2), x)
        self.disp2_up = self.crop_top_left(torch.nn.functional.interpolate(self.disp2,
                                                                      scale_factor=2,
                                                                      mode='bilinear',
                                                                      align_corners=False), x)
        self.concat1 = torch.cat((self.out_upconv1, self.disp2_up), 1)
        self.out_iconv1 = self.iconv1(self.concat1)
        disp1 = self.alpha * self.predict_disp1(self.out_iconv1) + self.beta

        if self.training:
            return disp1, self.disp2
        else:
            return disp1


if __name__ == "__main__":
    net = LFDispNetViz(in_channels=8, out_channels=8)
    net.eval()
    data = torch.rand(4, 8, 128, 128)
    y = net(data)
    print(y.shape)
