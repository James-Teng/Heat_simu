# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023.9.20 14:27
# @Author  : James.T
# @File    : datasets.py
import logging

import _init_cwd  # change cwd

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

region_casing_path = r'./data/region/region_casing.npy'
region_supervised_path = r'./data/region/region_supervised.npy'
region_data_path = r'./data/region/region_data.npy'
region_outer_path = r'./data/region/region_outer.npy'


class DatasetFromFolder(Dataset):
    """
    Heat simulation Dataset
    return frame t, t+n and mask
    """
    def __init__(
            self,
            image_list_paths: list[str],  # 需要提供数据列表所在的位置
            gaps: Optional[list[float]] = None,
            supervised_range: int = 1,
            transform_input: Optional[Callable] = None,
            transform_target: Optional[Callable] = None,
            transform_region: Optional[Callable] = None,
    ):
        """
        initialization

        :param image_list_paths: dataset folders
        :param transform_input: transforms applied to input
        :param transform_target: transforms applied to target
        :param transform_region: transforms applied to mask
        :returns: None
        """
        # self.data_folder_list = roots  # 训练需要包含的数据
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.transform_region = transform_region
        self.image_list = []
        self.gaps = gaps

        # 校验范围是否合法
        assert supervised_range >= 0, "illegal range"
        self.supervised_range = supervised_range

        lengths = [0]
        # 创建数据列表
        for data_list in image_list_paths:
            assert os.path.isfile(data_list) and data_list.endswith('.json'), \
                f"{self.__class__.__name__}: \'{data_list}\' is not a JSON file"
            with open(data_list, 'r') as jsonfile:
                tmp = json.load(jsonfile)
                lengths.append(lengths[-1] + len(tmp))
                self.image_list.extend(tmp)

        lengths.pop(0)
        self.index_limits = [ll - self.supervised_range - 1 for ll in lengths]

        self.length = len(self.image_list) - len(self.index_limits) * self.supervised_range

        # 加载区域
        self.region_casing = np.load(region_casing_path)
        self.region_supervised = np.load(region_supervised_path)
        self.region_data = np.load(region_data_path)
        self.region_outer = np.load(region_outer_path)

    def __getitem__(self, idx):
        """
        get input distributions, supervised distributions and masks
        when transforms applied, the output is tensor, otherwise is list of numpy array
        :return in_distribs, out_distribs, mask
        """

        # index mapping
        midx = idx
        choose_gap = 0
        for limit in self.index_limits:
            if midx > limit:
                midx += self.supervised_range
                choose_gap += 1
            else:
                break

        # load
        distribs = []
        for i in range(self.supervised_range + 1):
            distribs.append(np.load(self.image_list[midx+i]))
            logging.debug(f'{self.image_list[midx+i]} loaded')
        logging.debug(f'load {self.supervised_range + 1} distributions')
        in_distribs = distribs[:-1]
        out_distribs = distribs[1:]
        del distribs

        # the sequence of random process must be the same
        # set seed to make sure crop at same position
        seed = torch.random.seed()

        # transforms to input
        if self.transform_input:
            for i in range(len(in_distribs)):
                torch.random.manual_seed(seed)
                in_distribs[i] = self.transform_input(in_distribs[i])
                torch.random.manual_seed(seed)
                in_distribs[i] = in_distribs[i] * self.transform_region(self.region_data)  # mask input where is valid
            # stack distributions
            in_distribs = torch.stack(in_distribs, dim=0)

        # transforms to target
        if self.transform_target and self.supervised_range > 0:
            for i in range(len(out_distribs)):
                torch.random.manual_seed(seed)
                out_distribs[i] = self.transform_target(out_distribs[i])  # todo 是否需要在这里给 y 添加 mask
            # stack distributions
            out_distribs = torch.stack(out_distribs, dim=0)

        # transforms to region
        if self.transform_region:
            torch.random.manual_seed(seed)
            region_casing = self.transform_region(self.region_casing * self.gaps[choose_gap])
            torch.random.manual_seed(seed)
            region_supervised = self.transform_region(self.region_supervised)
            torch.random.manual_seed(seed)
            region_data = self.transform_region(self.region_data)
            torch.random.manual_seed(seed)
            region_outer = self.transform_region(self.region_outer)
        else:
            region_supervised = self.region_supervised
            region_casing = self.region_casing
            region_data = self.region_data
            region_outer = self.region_outer

        return in_distribs, out_distribs, region_casing, region_supervised, region_data, region_outer

    def __len__(self):
        """
        the quantity of images
        """
        return self.length


# todo 换成类，继承自 datasetfromfolder，添加一些统计功能
def SimuHeatDataset(
        roots: list[str],  # 需要包含的训练数据
        gaps: list[float],  # 上述数据其对应的 热阻
        time_intervals: list[str],  # 期望的时间间隔，返回相同数量的数据集
        supervised_range: int = 1,
        flip: bool = True,
        crop_size: Optional[int] = None,
) -> dict:
    """
    build datasets, and compose transforms

    :param roots: dataset roots
    :param gaps:
    :param time_intervals:
    :param supervised_range:
    :param flip:
    :param crop_size:

    :returns:
        the configured dataset
    """

    # transforms
    input_trans, mask_trans, target_trans = utils.compose_transforms(crop=crop_size, flip=flip)

    # make datasets
    simu_heat_datasets = {}
    for interv in time_intervals:
        image_list_paths = [
            os.path.join(r, f'data_list_interval_{interv}.json')
            for r in roots
            if os.path.isfile(os.path.join(r, f'data_list_interval_{interv}.json'))
        ]
        simu_heat_datasets[interv] = DatasetFromFolder(
            image_list_paths=image_list_paths,
            gaps=gaps,
            supervised_range=supervised_range,
            transform_input=input_trans,
            transform_target=target_trans,
            transform_region=mask_trans,
        )

    return simu_heat_datasets


def cat_input(distribution, region):
    return torch.cat([distribution, region], dim=1)


if __name__ == '__main__':

    # ------ debugging ------ #

    logging.basicConfig(level=logging.WARNING)
    # test
    test_dataset = DatasetFromFolder(
        [
            r'./data/data3_gap/tensor_format_2interval/gap0.1/data_list_interval_1000.0.json',
        ],
        gaps=[0.1],
        supervised_range=10,
        transform_input=utils.compose_input_transforms(),
        transform_region=utils.compose_mask_transforms(),
        transform_target=utils.compose_target_transforms(),
    )
    print('dataset length: ', len(test_dataset))

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=5,
        shuffle=True,
    )

    x, y, casing, supervised, data, outer = next(iter(test_dataloader))
    for d in x[0]:
        print(d)
    # print('target range', len(y))
    print('input shape', x.shape)
    print('target shape', y.shape)
    print('casing shape', casing.shape)
    print('mask shape', supervised.shape)
    print('mask grad', supervised.requires_grad)
    print('cat shape', cat_input(x[:, 0, :, :, :], casing).shape)

    # # calculate mean and std
    # print(utils.get_stat(td, 1))

    # plt.figure()
    # plt.imshow(dis[1].numpy().transpose(1, 2, 0), vmin=-1, vmax=1, cmap='jet')
    # plt.axis('off')
    # plt.show()
