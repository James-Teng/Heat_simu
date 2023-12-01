# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023.9.23 13:37
# @Author  : James.T
# @File    : arch.py

import _init_cwd  # change cwd

from typing import Optional

import torch
from torch import nn

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


# todo 多步监督, 这个应该直接写到模型里面去
# todo RNN + hourglass 结构
# todo 单独使用一个卷积来得到最后的内壳温度分布
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


# todo 加入 dropout
class SimpleArchR(nn.Module):
    """

    """
    def __init__(
            self,
            large_kernel_size: int = 9,
            small_kernel_size: int = 3,
            in_channels: int = 1,
            out_channels: int = 1,
            n_channels: int = 32,
            n_blocks: int = 2,
            activation: str = 'PReLU',
    ):
        super().__init__()

        # conv_k9n64s1 PReLU
        self.conv_block1 = ConvolutionalBlock(
            in_channels=in_channels,
            out_channels=n_channels,
            kernel_size=large_kernel_size,
            is_bn=False,
            activation=activation,
        )

        # residual blocks
        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(
                    kernel_size=small_kernel_size,
                    channels=n_channels,
                    pre_activate=False,
                    is_bn=True,
                )
                for i in range(n_blocks)
            ]
        )

        # conv_k3n64s1  bn  s1
        self.conv_block2 = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=small_kernel_size,
            is_bn=True,
            activation=None,
        )

        # conv_k9n3s1
        self.conv_block3 = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=out_channels,
            kernel_size=large_kernel_size,
            is_bn=False,
            activation='tanh',
        )

    def forward(self, x):
        residual_output = self.conv_block1(x)
        output = self.residual_blocks(residual_output)
        output = self.conv_block2(output)
        output = output + residual_output
        output = self.conv_block3(output)
        return output


if __name__ == '__main__':

    # test
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # # ConvolutionalBlock
    # ta = ConvolutionalBlock(10, 20, 3, 1, 'relu', True, True)
    # ta = ta.to(device)
    # summary(ta, input_size=[(10, 96, 96)])

    # # Residual Block
    # ta = ResidualBlock(activation='relu', pre_activate=False)
    # ta = ta.to(device)
    # summary(ta, input_size=[(64, 24, 24)])
    # # rand_tensor = torch.randn([1, 64, 24, 24]).to(device)
    # # out = ta(rand_tensor)
    # # l = torch.mean(out)
    # # l.backward()

    # SimpleArchR
    ta = SimpleArchR()
    ta = ta.to(device)
    summary(ta, input_size=[(1, 260, 130)])


