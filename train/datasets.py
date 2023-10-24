# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023.9.20 14:27
# @Author  : James.T
# @File    : datasets.py

import os
import json
from collections.abc import Callable
from typing import Optional

import torch
from torch.utils.data import Dataset

import numpy as np
from torchvision import transforms
import torchvision

import matplotlib.pyplot as plt

import utils


# to do:
# 是否需要将 input 的 transform 改成和 target 是一样的，然后再写一个 target to input 的 trans，这样循环过程会更一致一些

region_casing_path = r'E:\Research\Project\Heat_simu\data\data2_even\tensor_format\region_casing.npy'
region_supervised_path = r'E:\Research\Project\Heat_simu\data\data2_even\tensor_format\region_supervised.npy'


class DatasetFromFolder(Dataset):
    """
    Heat simulation Dataset
    return frame t, t+n and mask
    """
    def __init__(
            self,
            roots: list[str],
            supervised_range: int = 1,
            transform_input: Optional[Callable] = None,
            transform_target: Optional[Callable] = None,
            transform_region: Optional[Callable] = None,
    ):
        """
        initialization

        :param root: dataset folder
        :param transform_input: transforms applied to input
        :param transform_target: transforms applied to target
        :param transform_mask: transforms applied to mask
        :returns: None
        """
        # self.data_folder_list = roots  # 训练需要包含的数据
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.transform_region = transform_region
        self.image_list = []

        # 校验范围是否合法
        assert supervised_range >= 0, "illegal range"
        self.supervised_range = supervised_range

        lengths = [0]
        # 创建数据列表
        for data_folder in roots:
            # assert os.path.exists(data_folder), "Folder doesn't exist"
            with open(os.path.join(data_folder, 'data_list.json'), 'r') as jsonfile:
                tmp = json.load(jsonfile)
                lengths.append(lengths[-1] + len(tmp))
                self.image_list.extend(tmp)

        lengths.pop(0)
        self.index_limits = [ll - self.supervised_range - 1 for ll in lengths]

        self.length = len(self.image_list) - len(self.index_limits) * self.supervised_range

        # 加载区域
        self.region_casing = np.load(region_casing_path)
        self.region_supervised = np.load(region_supervised_path)


    def __getitem__(self, idx):
        """
        get distributions and mask
        :return x, y, mask
        """

        # index mapping
        midx = idx
        for limit in self.index_limits:
            if midx > limit:
                midx += self.supervised_range
            else:
                break

        # load
        distribs = []
        for i in range(self.supervised_range + 1):
            distribs.append(np.load(self.image_list[midx]))

        # transforms 随机过程的顺序需要一致
        # set seed to make sure crop at same position
        seed = torch.random.seed()

        # transforms to input
        distribs[0] = np.stack([distribs[0], self.region_casing], axis=2)  # 扩维拼接通道，加入了损伤后需要修改
        if self.transform_input:
            torch.random.manual_seed(seed)
            distribs[0] = self.transform_input(distribs[0])

        # transforms to target
        if self.transform_target and self.supervised_range > 0:
            for i in range(1, self.supervised_range + 1):
                torch.random.manual_seed(seed)
                distribs[i] = self.transform_target(distribs[i])

        # transforms to region
        if self.transform_region:
            torch.random.manual_seed(seed)
            region_supervised = self.transform_region(self.region_supervised)
        else:
            region_supervised = self.region_supervised

        return distribs[0], distribs[1:], region_supervised  # 需要自定义 collate_fn, 主要是 target

    def __len__(self):
        """
        the quantity of images
        """
        return self.length


# todo 是否需要这一层的封装？目前来说需要区分 是否为训练 是否裁切 是否旋转 监督范围
def SimuHeatDataset(
        root: str,
        train: bool,
        rotate: bool,
        supervised_range: int,
        crop_size: Optional[int] = None,
) -> DatasetFromFolder:
    """
    build a dataset, and compose transforms

    :param root:
        dataset root

    :returns:
        the configured dataset
    """

    pass


if __name__ == '__main__':

    # ------ debugging ------ #

    # # test for DatasetFromFolder
    # td = DatasetFromFolder(
    #     r'E:\Research\Project\Heat_simu\data\data2_even\tensor_format\0.1K_0.1gap',
    #     supervised_range=1,
    #     transform_input=utils.compose_input_transforms(),
    #     transform_mask=utils.compose_mask_transforms(),
    #     transform_target=utils.compose_target_transforms(),
    # )
    # print('dataset length: ', len(td))
    # dis, mask = td[189]
    #
    # dis[1] = dis[1] * mask
    #
    # print(dis[1].max(), dis[1].min())
    # plt.figure()
    # plt.imshow(dis[1].numpy().transpose(1, 2, 0), vmin=-1, vmax=1, cmap='jet')
    # plt.axis('off')
    # plt.show()

    # # calculate mean and std
    # print(utils.get_stat(td, 1))

    # test for collater
    test_dataset = DatasetFromFolder(
        [
            r'E:\Research\Project\Heat_simu\data\data2_even\tensor_format\0.1K_0.1gap',
        ],
        supervised_range=4,
        transform_input=utils.compose_input_transforms(),
        transform_region=utils.compose_mask_transforms(),
        transform_target=utils.compose_target_transforms(),
    )
    print('dataset length: ', len(test_dataset))

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
    )

    x, y, mask = next(iter(test_dataloader))
    print('target range', len(y))
    print('input shape', x.shape)
    print('target shape', y[0].shape)
    print('mask shape', mask.shape)

    # plt.figure()
    # plt.imshow(dis[1].numpy().transpose(1, 2, 0), vmin=-1, vmax=1, cmap='jet')
    # plt.axis('off')
    # plt.show()



