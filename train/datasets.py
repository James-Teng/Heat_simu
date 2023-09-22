# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023.9.20 14:27
# @Author  : James.T
# @File    : datasets.py

import os
import utils
from collections.abc import Callable
from typing import Optional

import torch
from torch.utils.data import Dataset

import numpy as np
from torchvision import transforms
import torchvision

import matplotlib.pyplot as plt


# to do:
# 是否需要将 input 的 transform 改成和 target 是一样的，然后再写一个 target to input 的 trans，这样循环过程会更一致一些

class DatasetFromFolder(Dataset):
    """
    Heat simulation Dataset
    return frame t, t+n and mask
    """
    def __init__(
            self,
            root: str,
            supervised_range: int = 1,
            transform_input: Optional[Callable] = None,
            transform_target: Optional[Callable] = None,
            transform_mask: Optional[Callable] = None,
    ):
        """
        initialization

        :param root: dataset folder
        :param transform_input: transforms applied to input
        :param transform_target: transforms applied to target
        :param transform_mask: transforms applied to mask
        :returns: None
        """
        self.data_folder = root  # 需要具体指定某个数据集的文件夹
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.transform_mask = transform_mask

        # 校验范围是否合法
        assert supervised_range >= 0, "illegal range"
        self.supervised_range = supervised_range

        # 校验地址是否存在
        assert os.path.exists(self.data_folder), "Folder doesn't exist"

        # 统计 len
        self.length = int(len(os.listdir(self.data_folder))/2 - 1) - self.supervised_range   # 鲁棒性不佳，需要再-1最后一张图没有监督对象

    def __getitem__(self, idx):
        """
        get distributions and mask
        distribs
        input index at 0
        """

        # load
        distribs = []
        for i in range(self.supervised_range + 1):
            distribs.append(
                np.load(
                    os.path.join(
                        self.data_folder,
                        f'{idx + i}.npy',
                    )
                )
            )
        mask = np.load(
            os.path.join(
                self.data_folder,
                'mask.npy'
            )
        )

        # transforms 随机过程的顺序需要一致
        # set seed to make sure crop at same position
        seed = torch.random.seed()

        # transforms to input
        if self.transform_input:
            torch.random.manual_seed(seed)
            distribs[0] = self.transform_input(distribs[0])

        # transforms to target
        if self.transform_target and self.supervised_range > 0:
            for i in range(1, self.supervised_range + 1):
                torch.random.manual_seed(seed)
                distribs[i] = self.transform_target(distribs[i])

        # transforms to mask
        if self.transform_mask:
            torch.random.manual_seed(seed)
            mask = self.transform_mask(mask)

        return distribs, mask  # 需要自定义 collate_fn, 主要是 target

    def __len__(self):
        """
        the quantity of images
        """
        return self.length


# 是否需要这一层的封装？目前来说需要区分 是否为训练 是否裁切 是否旋转 监督范围
def SimuHeatDataset(
        root: str,
        train: bool,  # 暂时没有这个功能
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

    # test for DatasetFromFolder
    td = DatasetFromFolder(
        r'E:\Research\Project\Heat_simu\data\data2_even\tensor_format\0.1K_0.1gap',
        supervised_range=0,
    )
    print(len(td))
    dis, mask = td[190]
    print(len(dis))
    plt.figure()
    plt.imshow(dis[0], vmin=0, vmax=400, cmap='jet')  # 建立颜色映射
    plt.axis('off')
    plt.figure()
    plt.imshow(mask, vmin=0, vmax=400, cmap='jet')  # 建立颜色映射
    plt.axis('off')
    plt.show()

    # calculate mean and std
    print(utils.get_stat(td, 1))

