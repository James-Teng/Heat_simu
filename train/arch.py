# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023.9.23 13:37
# @Author  : James.T
# @File    : arch.py

import pp  # change cwd

from typing import Optional
import logging

import torch
from torch import nn
from torchinfo import summary

import utils

# todo 加载权重检查

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


# todo hourglass 结构
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


class SimpleExtractor(nn.Module):
    def __init__(
            self,
            in_channels: int = 2,
            out_channels: int = 32,
            kernel_size: int = 9,
            activation: str = 'PReLU',
            is_bn: bool = False,
    ):
        super().__init__()
        self.conv_block = ConvolutionalBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            is_bn=False,
            activation=activation,
        )

    def forward(self, x):
        y = self.conv_block(x)
        return y


class SimpleBackbone(nn.Module):
    def __init__(
            self,
            kernel_size: int = 3,
            n_channels: int = 32,
            n_blocks: int = 4,
            is_bn: bool = True,
    ):
        super().__init__()
        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(
                    kernel_size=kernel_size,
                    channels=n_channels,
                    pre_activate=False,
                    is_bn=is_bn,
                )
                for i in range(n_blocks)
            ]
        )
        self.conv_block = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            is_bn=is_bn,
            activation=None,
        )

    def forward(self, x):
        y = self.residual_blocks(x)
        y = self.conv_block(y)
        return y + x


# todo 单独使用一个卷积来得到最后的内壳温度分布
class SimpleRegressor(nn.Module):
    def __init__(
            self,
            kernel_size: int = 9,
            in_channels: int = 32,
            out_channels: int = 1,
            activation: str = 'tanh',
    ):
        super().__init__()
        self.conv_block = ConvolutionalBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            is_bn=False,
            activation=activation,
        )

    def forward(self, x):
        y = self.conv_block(x)
        return y


# todo 加入 dropout
class SimpleArchR(nn.Module):
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

        # conv_k9n32s1 PReLU
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

        # conv_k3n32s1  bn  s1
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


class NaiveRNNFramework(nn.Module):
    """
    see framework.pptx in docs
    """
    def __init__(
            self,
            extractor: nn.Module,
            regressor: nn.Module,
            backbone: nn.Module,
            out2intrans,
    ):
        """
        initialization of NaiveRNNFramework
        :param extractor: feature extractor
        :param regressor: distribution regressor
        :param backbone: backbone model
        :param out2intrans: transforms from output to input
        """
        super().__init__()
        self.extractor = extractor
        self.regressor = regressor
        self.backbone = backbone
        self.out2intrans = out2intrans

        # self.region_casing = None
        # self.region_supervised = None
        # self.region_data = None
        # self.region_outer = None

        self.is_interval_output = True

    def forward(self, x, region_casing, region_supervised, region_data, region_outer):
        """
        forward
        input x: (batch_size, time_step, channel, height, width)
        :param x: input is a sequence of heat distribution
        :return: sequence of heat distribution or final heat distribution
        """

        # if no batch dim, add one
        if len(x.shape) != 5:
            x.unsqueeze_(dim=0)

        output = []
        distribution = x[:, 0, :, :, :]
        # time_steps = x.shape[1]
        for i in range(x.shape[1]):

            # set outer distribution and cat channels
            # todo 后续可以调制这里的输入，研究怎么表达热阻，也许可以在regressor中乘上mask一个热阻系数，让模型去调制升温的多少
            logging.debug(f'distribution: {distribution.shape}')
            logging.debug(f'region_supervised: {region_supervised.shape}')
            logging.debug(f'region_outer: {region_outer.shape}')
            logging.debug(f'distribution requires grad: {distribution.requires_grad}')
            distribution = distribution * region_supervised + x[:, i, :, :, :] * region_outer
            input_x = torch.cat([distribution, region_casing], dim=1)

            # extract features
            if i == 0:
                features = self.extractor(input_x)
            else:
                features = self.extractor(input_x) + features  # todo 思考直接相加融合特征的合理性

            # backbone
            features = self.backbone(features)
            logging.debug(f'features: {features.shape}')

            # regression
            distribution = self.regressor(features) * region_supervised

            # interval and final output
            if self.is_interval_output or i == x.shape[1] - 1:
                output.append(distribution)

            # out to in
            if i < x.shape[1] - 1:
                distribution = self.out2intrans(distribution)

        if self.is_interval_output:
            return torch.stack(output, dim=1)
        else:
            return output[-1]

    # def set_region(self, region_casing, region_supervised, region_data, region_outer):
    #     self.region_casing = region_casing
    #     self.region_supervised = region_supervised
    #     self.region_data = region_data
    #     self.region_outer = region_outer

    def enable_interval_output(self):
        """
        enable interval output
        :return:
        """
        self.is_interval_output = True

    def disable_interval_output(self):
        """
        disable interval output
        :return:
        """
        self.is_interval_output = False


class IterativeFramework(nn.Module):
    """
    see framework.pptx in docs
    """
    def __init__(
            self,
            extractor: nn.Module,
            regressor: nn.Module,
            backbone: nn.Module,
            out2intrans,
    ):
        """
        initialization of NaiveRNNFramework
        :param extractor: feature extractor
        :param regressor: distribution regressor
        :param backbone: backbone model
        :param out2intrans: transforms from output to input
        """
        super().__init__()
        self.extractor = extractor
        self.regressor = regressor
        self.backbone = backbone
        self.out2intrans = out2intrans

        self.is_interval_output = True

    def forward(self, x, region_casing, region_supervised, region_data, region_outer):
        """
        forward
        input x: (batch_size, time_step, channel, height, width)
        :param region_outer:
        :param region_data:
        :param region_supervised:
        :param region_casing:
        :param x: input is a sequence of heat distribution
        :return: sequence of heat distribution or final heat distribution
        """

        # if no batch dim, add one
        if len(x.shape) != 5:
            x.unsqueeze_(dim=0)

        output = []
        distribution = x[:, 0, :, :, :]
        # time_steps = x.shape[1]
        for i in range(x.shape[1]):

            # set outer distribution and cat channels
            distribution = distribution * region_supervised + x[:, i, :, :, :] * region_outer
            input_x = torch.cat([distribution, region_casing], dim=1)

            # extract features
            distribution = self.regressor(self.backbone(self.extractor(input_x))) * region_supervised

            # interval and final output
            if self.is_interval_output or i == x.shape[1] - 1:
                output.append(distribution)

            # out to in
            if i < x.shape[1] - 1:
                distribution = self.out2intrans(distribution)

        if self.is_interval_output:
            return torch.stack(output, dim=1)
        else:
            return output[-1]

    def enable_interval_output(self):
        """
        enable interval output
        :return:
        """
        self.is_interval_output = True

    def disable_interval_output(self):
        """
        disable interval output
        :return:
        """
        self.is_interval_output = False


def model_factory(config):
    """
    model factory
    :param config: config dict
    :return: model
    """
    # todo 是否有多余的参数
    keys = [
        'model_type', 'in_channels', 'n_channels',
        'large_kernel_size', 'small_kernel_size',
        'blocks', 'bn'
    ]

    neccessary_config = {key: config[key] for key in keys}

    if neccessary_config['model_type'] == 'SimpleArchR':
        needed_keys = ['large_kernel_size', 'small_kernel_size', 'n_channels', 'blocks', 'in_channels']
        assert all([neccessary_config[key] is not None for key in needed_keys]), \
            f'missing keys for {neccessary_config["model_type"]}'
        return SimpleArchR(
            large_kernel_size=config['large_kernel_size'],
            small_kernel_size=config['small_kernel_size'],
            in_channels=config['in_channels'],
            n_channels=config['n_channels'],
            n_blocks=config['blocks'],
        )

    elif neccessary_config['model_type'] == 'NaiveRNNFramework':
        needed_keys = ['large_kernel_size', 'small_kernel_size', 'n_channels', 'blocks', 'in_channels', 'bn']
        assert all([neccessary_config[key] is not None for key in needed_keys]), \
            f'missing keys for {neccessary_config["model_type"]}'
        return NaiveRNNFramework(
            extractor=SimpleExtractor(
                in_channels=neccessary_config['in_channels'],
                out_channels=neccessary_config['n_channels'],
                kernel_size=neccessary_config['large_kernel_size'],
            ),
            backbone=SimpleBackbone(
                n_blocks=neccessary_config['blocks'],
                n_channels=neccessary_config['n_channels'],
                kernel_size=neccessary_config['small_kernel_size'],
                is_bn=neccessary_config['bn'],
            ),
            regressor=SimpleRegressor(
                kernel_size=neccessary_config['large_kernel_size'],
                in_channels=neccessary_config['n_channels'],
                out_channels=1,
            ),
            out2intrans=utils.compose_target2input_transforms()
        )

    elif neccessary_config['model_type'] == 'IterativeFramework':
        needed_keys = ['large_kernel_size', 'small_kernel_size', 'n_channels', 'blocks', 'in_channels', 'bn']
        assert all([neccessary_config[key] is not None for key in needed_keys]), \
            f'missing keys for {neccessary_config["model_type"]}'
        return IterativeFramework(
            extractor=SimpleExtractor(
                in_channels=neccessary_config['in_channels'],
                out_channels=neccessary_config['n_channels'],
                kernel_size=neccessary_config['large_kernel_size'],
            ),
            backbone=SimpleBackbone(
                n_blocks=neccessary_config['blocks'],
                n_channels=neccessary_config['n_channels'],
                kernel_size=neccessary_config['small_kernel_size'],
                is_bn=neccessary_config['bn'],
            ),
            regressor=SimpleRegressor(
                kernel_size=neccessary_config['large_kernel_size'],
                in_channels=neccessary_config['n_channels'],
                out_channels=1,
            ),
            out2intrans=utils.compose_target2input_transforms()
        )

    else:
        raise ValueError(f'unsupported model type: \'{neccessary_config["model_type"]}\'')

# todo 模型配置器，配置表，参考human pose estimation
# todo 模型初始化表，在其中可以加载 weight


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    # test
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # NaiveRNNFramework
    ta = NaiveRNNFramework(
        extractor=SimpleExtractor(),
        backbone=SimpleBackbone(n_blocks=1),
        regressor=SimpleRegressor(),
        out2intrans=utils.compose_target2input_transforms()
    )
    ta.enable_interval_output()
    ta = ta.to(device)

    # in_rand = torch.randn(5, 2, 1, 260, 130).to(device)
    # mask_rand = torch.randn(5, 1, 260, 130).to(device)
    # out = ta(in_rand, mask_rand, mask_rand, mask_rand, mask_rand)

    summary(ta, input_size=[(5, 2, 1, 260, 130), (5, 1, 260, 130), (5, 1, 260, 130), (5, 1, 260, 130), (5, 1, 260, 130)])

    # simple_extractor = SimpleExtractor()
    # simple_extractor = simple_extractor.to(device)
    # summary(simple_extractor, input_size=(2, 260, 130), batch_dim=0)

