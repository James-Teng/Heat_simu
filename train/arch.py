# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023.9.23 13:37
# @Author  : James.T
# @File    : arch.py


import torch
from torch import nn
from torchvision.models import vgg19, VGG19_Weights, resnet34, ResNet34_Weights
import torchvision

from typing import Optional

from torchsummary import summary


class ConvolutionalBlock(nn.Module):
    """
    Convolutional Block, Consists of Convolution, BatchNorm, Activate function
    support pre activation
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            activation: Optional[str] = None,
            is_bn: bool = False,
            pre_activate: bool = False,
    ):
        super().__init__()

        if activation:
            activation = activation.lower()
            assert activation in {'prelu', 'relu', 'leakyrelu', 'tanh'}, 'no matching activation type'

        # bn
        layers = []
        if is_bn:
            layers.append(
                nn.BatchNorm2d(
                    num_features=in_channels if pre_activate else out_channels
                )
            )

        # activation
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        # conv
        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False if is_bn else True  # 有 bn 的时候不需要 bias
        )

        # pre activate
        if pre_activate:
            layers.append(conv_layer)
        else:
            layers.insert(0, conv_layer)

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv_block(x)
        return y


# todo 考虑优化残差模块，让跳接线路上没有模块的阻挡
class ResidualBlock(nn.Module):
    """
    不需要 down sample
    和 resnet 的 block 有一些不同，最后没有 activate
    SRGAN 的 Residual Block
    """
    def __init__(
            self,
            kernel_size: int = 3,
            channels: int = 64,  # 需要调整
            activation: str = 'prelu',
            pre_activate: bool = False,
            is_bn: bool = True,
    ):
        super().__init__()
        self.two_conv_block = nn.Sequential(
            ConvolutionalBlock(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                is_bn=is_bn,
                activation=activation,
                pre_activate=pre_activate
            ),
            ConvolutionalBlock(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                is_bn=is_bn,
                activation=activation if pre_activate else None,
                pre_activate=pre_activate
            ),
        )

    def forward(self, x):
        y = self.two_conv_block(x)
        return y + x


# class SRResNet(nn.Module):
#     """
#
#     """
#     def __init__(
#             self,
#             large_kernel_size: int = 9,
#             small_kernel_size: int = 3,
#             n_channels: int = 64,
#             n_blocks: int = 16,
#             scaling_factor: int = 4,
#     ):
#         super().__init__()
#         assert scaling_factor in {2, 4, 8}, 'scaling_factor not allowed, should be 2, 4 or 8'
#
#         # conv_k9n64s1 PReLU
#         self.conv_block1 = ConvolutionalBlock(
#             in_channels=3,
#             out_channels=n_channels,
#             kernel_size=large_kernel_size,
#             is_bn=False,
#             activation='PReLu',
#         )
#
#         # residual blocks
#         self.residual_blocks = nn.Sequential(
#             *[
#                 ResidualBlock(
#                     kernel_size=small_kernel_size,
#                     channels=n_channels,
#                 )
#                 for i in range(n_blocks)
#             ]
#         )
#
#         # conv_k3n64s1  bn  s1
#         self.conv_block2 = ConvolutionalBlock(
#             in_channels=n_channels,
#             out_channels=n_channels,
#             kernel_size=small_kernel_size,
#             is_bn=True,
#             activation=None,
#         )
#
#         # sub pixel shuffle blocks
#         n_sub_blocks = int(math.log2(scaling_factor))
#         self.sub_pixel_shuffle_blocks = nn.Sequential(
#             *[
#                 SubPixelConvolutionalBlock(
#                     channels=n_channels,
#                     kernel_size=small_kernel_size,
#                     scaling_factor=2,
#                 )
#                 for i in range(n_sub_blocks)
#             ]
#         )
#
#         # conv_k9n3s1
#         self.conv_block3 = ConvolutionalBlock(
#             in_channels=n_channels,
#             out_channels=3,
#             kernel_size=large_kernel_size,
#             is_bn=False,
#             activation='tanh',
#         )
#
#     def forward(self, x):
#         residual_output = self.conv_block1(x)
#         output = self.residual_blocks(residual_output)
#         output = self.conv_block2(output)
#         output = output + residual_output
#         output = self.sub_pixel_shuffle_blocks(output)
#         output = self.conv_block3(output)
#         return output
#
#
# class SRGenerator(nn.Module):
#     """
#
#     """
#     def __init__(
#             self,
#             large_kernel_size: int = 9,
#             small_kernel_size: int = 3,
#             n_channels: int = 64,
#             n_blocks: int = 16,
#             scaling_factor: int = 4
#     ):
#         """
#
#         """
#         super().__init__()
#         self.srresnet = SRResNet(
#             large_kernel_size=large_kernel_size,
#             small_kernel_size=small_kernel_size,
#             n_channels=n_channels,
#             n_blocks=n_blocks,
#             scaling_factor=scaling_factor,
#         )
#
#     def forward(self, x):
#         output = self.srresnet(x)
#         return output
#
#
# class SRDiscriminator(nn.Module):
#     """
#
#     """
#     def __init__(
#             self,
#             kernel_size: int = 3,
#             n_channels: int = 64,
#             n_blocks: int = 8,
#             fc_size: int = 1024,
#     ):
#         """
#
#         """
#         super().__init__()
#
#         conv_blocks = []
#
#         # the first n_blocks
#         conv_blocks.append(
#             ConvolutionalBlock(
#                 in_channels=3,
#                 out_channels=n_channels,
#                 kernel_size=kernel_size,
#                 stride=1,
#                 is_bn=False,
#                 activation='leakyrelu',
#             )
#         )
#
#         # the rest conv block with bn
#         in_channel_build = n_channels
#         for i in range(n_blocks - 1):
#
#             # blocks that do down sample and channel up
#             if i % 2 == 0:
#                 out_channel_build = in_channel_build
#                 conv_blocks.append(
#                     ConvolutionalBlock(
#                         in_channels=in_channel_build,
#                         out_channels=out_channel_build,
#                         kernel_size=kernel_size,
#                         stride=2,
#                         is_bn=True,
#                         activation='leakyrelu',
#                     )
#                 )
#
#             # blocks that change nothing in shape
#             else:
#                 out_channel_build = in_channel_build * 2
#                 conv_blocks.append(
#                     ConvolutionalBlock(
#                         in_channels=in_channel_build,
#                         out_channels=out_channel_build,
#                         kernel_size=kernel_size,
#                         stride=1,
#                         is_bn=True,
#                         activation='leakyrelu',
#                     )
#                 )
#                 in_channel_build = in_channel_build * 2
#
#         self.conv_blocks = nn.Sequential(*conv_blocks)
#
#         # fix output feature size, 96 / 2**4 = 6
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
#         self.flatten = nn.Flatten()
#
#         # Dense 1024
#         self.fc1 = nn.Linear(out_channel_build * 6 * 6, fc_size)
#         self.leaky_relu = nn.LeakyReLU(0.2)
#
#         # Dense 1, no sigmoid, included in BCEWithLogitsLoss
#         self.fc2 = nn.Linear(fc_size, 1)
#
#     def forward(self, x):
#         y = self.conv_blocks(x)
#         y = self.adaptive_pool(y)
#         y = self.flatten(y)
#         y = self.fc1(y)
#         y = self.leaky_relu(y)
#         y = self.fc2(y)
#         return y


if __name__ == '__main__':

    # test
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # # ConvolutionalBlock
    # ta = ConvolutionalBlock(10, 20, 3, 1, 'relu', True, True)
    # ta = ta.to(device)
    # summary(ta, input_size=[(10, 96, 96)])

    # Residual Block
    ta = ResidualBlock(activation='relu', pre_activate=False)
    ta = ta.to(device)
    summary(ta, input_size=[(64, 24, 24)])
    # rand_tensor = torch.randn([1, 64, 24, 24]).to(device)
    # out = ta(rand_tensor)
    # l = torch.mean(out)
    # l.backward()

    # # SRResNet
    # ta = SRGenerator()
    # ta = ta.to(device)
    # summary(ta, input_size=[(3, 24, 24)])

    # # SRDis
    # ta = SRDiscriminator()
    # ta = ta.to(device)
    # summary(ta, input_size=[(3, 96, 96)])

    # # Truncated VGG19
    # ta = TruncatedVGG19(5, 4)
    # ta = ta.to(device)
    # summary(ta, input_size=[(3, 96, 96)])

    # vgg19_pretrained = vgg19(weights=VGG19_Weights.DEFAULT)
    # print(list(vgg19_pretrained.features.named_children()))


