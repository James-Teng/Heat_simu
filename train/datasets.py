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
        assert supervised_range >= 1, "illegal range"
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
        for i in range(self.supervised_range):
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
        if self.transform_target:
            for i in range(1, self.supervised_range):
                torch.random.manual_seed(seed)
                distribs[i] = self.transform_target(distribs[i])

        # transforms to mask
        if self.transform_mask:
            torch.random.manual_seed(seed)
            mask = self.transform_mask(mask)

        return distribs, mask

    def __len__(self):
        """
        the quantity of images
        """
        return self.length




# def SRDataset(
#         dataset_name: str,
#         lr_target_type: Optional[str] = None,
#         hr_target_type: Optional[str] = None,
#         crop_size: Optional[int] = None,
#         scaling_factor: Optional[int] = None,
# ) -> DatasetFromFolder:
#     """
#     build a dataset
#
#     :param dataset_name:
#         the name of the dataset, according to the name, get the path
#     :param lr_target_type:
#
#     :param hr_target_type:
#
#     :param crop_size:
#         determine the dataset is used fot training or not.
#         when True, the output HR and LR image will be cropped
#         the output HR image size
#     :param scaling_factor:
#         down sample HR image by scaling_factor
#
#     :returns:
#         the configured dataset
#     """
#     # 数据集名字 到 路径的 dict 映射，检查数据集是否存在
#     datasets_dict = {
#         'Set5': './data/Set5',
#         'Set14': './data/Set14',
#         'BSD100': './data/BSD100',
#         'COCO2014': './data/COCO2014',
#         'DF2K_OST': './data/DF2K_OST',
#         'VOC2012': './data/VOC2012',
#     }
#     assert dataset_name in datasets_dict.keys(), f'{dataset_name} doesnt exist'
#     dataset_path = datasets_dict[dataset_name]
#
#     # img_type
#     img_type_list = ['imagenet-norm', '[-1, 1]']
#     assert lr_target_type in img_type_list, f'no image type named {lr_target_type}'
#     assert hr_target_type in img_type_list, f'no image type named {hr_target_type}'
#
#     # 检查 crop 后的图片是否能被整除下采样
#     is_crop = bool(crop_size)
#     if is_crop:
#         assert crop_size % scaling_factor == 0, '剪裁尺寸不能被放大比整除'
#
#     # 检查 json 是否存在，没有就创建
#     if not os.path.isfile(os.path.join(dataset_path, img_list_file_name)):
#         print('Dataset: image_list doesnt exist, creating...')
#         create_data_list(dataset_path, crop_size)
#
#     # 选择 crop 的形式
#     if is_crop:
#         prep = transforms.RandomCrop(crop_size)
#     else:
#         prep = utils.ScalableCrop(scaling_factor=scaling_factor)
#
#     # lr 变换
#     if lr_target_type:
#         trans_lr = utils.compose_lr_transforms(
#             img_type=lr_target_type,
#             scaling_factor=scaling_factor
#         )
#     else:
#         trans_lr = None
#
#     # hr 变换
#     if hr_target_type:
#         trans_hr = utils.compose_hr_transforms(
#             img_type=hr_target_type
#         )
#     else:
#         trans_hr = None
#
#     dataset = DatasetFromFolder(
#         root=dataset_path,
#         transform_prep=prep,
#         transform_hr=trans_hr,
#         transform_lr=trans_lr,
#     )
#     return dataset
#
#
# def create_data_list(data_folder: str, min_size: int):
#     """
#     make a json-style list file which consists all image paths in data_folder.
#     and filter images smaller than min_size when is_train is True.
#
#     :param data_folder: dataset folder
#     :param min_size: the smallest acceptable image size
#     :param is_train: When True, filter min_size
#
#     :returns: None
#     """
#     image_list = []
#     is_crop = bool(min_size)
#     for root, dirs, files in os.walk(data_folder): # 可用 遍历 栈实现，但是用 os.walk()
#         for name in files:
#             _, filetype = os.path.splitext(name)
#             if not filetype.lower() in ['.jpg', '.bmp', '.png']:
#                 continue
#             img_path = os.path.join(root, name)
#             if is_crop:
#                 img = Image.open(img_path, mode='r')
#                 if not (img.width >= min_size and img.height >= min_size):
#                     continue
#             image_list.append(img_path)
#     with open(os.path.join(data_folder, img_list_file_name), 'w') as jsonfile:
#         json.dump(image_list, jsonfile)


if __name__ == '__main__':

    # debugging

    # test for DatasetFromFolder
    td = DatasetFromFolder(r'E:\Research\Project\Heat_simu\data\data2_even\tensor_format\0.1K_0.1gap')
    dis, mask = td[0]
    plt.figure()
    plt.imshow(dis[0], vmin=0, vmax=400, cmap='jet')  # 建立颜色映射
    plt.axis('off')
    plt.figure()
    plt.imshow(mask, vmin=0, vmax=400, cmap='jet')  # 建立颜色映射
    plt.axis('off')
    plt.show()

