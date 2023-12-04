# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023.9.20 14:27
# @Author  : James.T
# @File    : datasets.py

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


# to do:
# 是否需要将 input 的 transform 改成和 target 是一样的，然后再写一个 target to input 的 trans，这样循环过程会更一致一些

region_casing_path = r'./data/data2_even/tensor_format/region_casing.npy'
region_supervised_path = r'./data/data2_even/tensor_format/region_supervised.npy'
region_data_path = r'./data/data2_even/tensor_format/region_data.npy'
region_outer_path = r'./data/data2_even/tensor_format/region_outer.npy'


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
            assert os.path.isfile(data_list), f"{data_list} is not a file"
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
        get distributions and mask
        :return x, y, mask
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

        # transforms 随机过程的顺序需要一致
        # set seed to make sure crop at same position
        seed = torch.random.seed()

        # transforms to input

        # distribs[0] = np.stack(
        #     [distribs[0], self.region_casing * self.gaps[choose_gap]], axis=2
        # )  # 扩维拼接通道，加入了损伤后需要修改

        if self.transform_input:
            torch.random.manual_seed(seed)
            distribs[0] = self.transform_input(distribs[0])
            torch.random.manual_seed(seed)
            distribs[0] = distribs[0] * self.transform_region(self.region_data)

        # transforms to target
        if self.transform_target and self.supervised_range > 0:
            for i in range(1, self.supervised_range + 1):
                torch.random.manual_seed(seed)
                distribs[i] = self.transform_target(distribs[i])

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

        return distribs[0], distribs[1:], region_casing, region_supervised, region_data, region_outer

    def __len__(self):
        """
        the quantity of images
        """
        return self.length


# todo 是否需要这一层的封装？目前来说需要区分 是否为训练 是否裁切 是否旋转 监督范围
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

    # test
    test_dataset = DatasetFromFolder(
        [
            r'./data/data3_gap/tensor_format_2interval/gap0.1/data_list_interval_1000.0.json',
        ],
        gaps=[0.1],
        supervised_range=4,
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
    # x[0][0] = x[0][0] * mask[0]
    print(x[0])
    print('target range', len(y))
    print('input shape', x.shape)
    print('target shape', y[0].shape)
    print('casing shape', casing.shape)
    print('mask shape', supervised.shape)
    print('mask grad', supervised.requires_grad)
    print('cat shape', cat_input(x, casing).shape)

    # # calculate mean and std
    # print(utils.get_stat(td, 1))

    # plt.figure()
    # plt.imshow(dis[1].numpy().transpose(1, 2, 0), vmin=-1, vmax=1, cmap='jet')
    # plt.axis('off')
    # plt.show()










    # # SimuheatDataset 使用实例
    # dataset_dict = SimuHeatDataset(
    #     time_intervals=[
    #         '1000.0',
    #         '10.0',
    #         '0.1',
    #     ],  # 指定时间间隔
    #     roots=[
    #         r'./data/data2_even/tensor_format/0.1K_0.1gap',  # 数据所在的文件夹
    #         r'./data/data2_even/tensor_format/0.1K_0.3gap',
    #         r'./data/data2_even/tensor_format/0.1K_0.5gap',
    #     ],
    #     gaps=[
    #         0.1,
    #         0.3,
    #         0.5,
    #     ],
    #     supervised_range=1,
    #     flip=True,
    #     crop_size=None
    # )
    #
    # print(dataset_dict)
    # print(len(dataset_dict['1000.0']))
    # print(len(dataset_dict['10.0']))
    # print(len(dataset_dict['0.1']))
