import sys
import time
import os
import json
from json.decoder import JSONDecodeError
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torchvision
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from collections.abc import Callable

# ----------------
#   statistics
# ----------------

# todo 换用只需要一次遍历的算法
def get_stat(train_dataset, channel):
    """
    Compute mean and std for training data
    :param train_dataset:
    :param channel:
    :return: (mean, std)
    """
    # 统计均值
    mean = np.zeros(channel)
    for i in range(len(train_dataset)):
        dis, _ = train_dataset[i]
        dis[0] = dis[0].reshape((channel, -1))
        for d in range(channel):
            data_except_0 = np.delete(dis[0][d, :], np.where(dis[0][d, :] == 0))
            mean[d] += data_except_0.mean()
    mean /= len(train_dataset)

    # 统计标准差
    squared_mean = np.zeros(channel)
    for i in range(len(train_dataset)):
        dis, _ = train_dataset[i]
        dis[0] = dis[0].reshape((channel, -1))
        for d in range(channel):
            data_except_0 = np.delete(dis[0][d, :], np.where(dis[0][d, :] == 0))
            squared_mean[d] += np.square(data_except_0 - mean[d]).mean()
    std = np.sqrt(squared_mean/len(train_dataset))

    return list(mean), list(std)

# ----------------
#   visualization
# ----------------


# --------------------
#   log
# --------------------
def save_img_to_file(imgs, path):
    torchvision.utils.save_image(imgs, path)


def plt_save_image(img, mask, img_path):
    """

    :param img: numpy 格式图片
    :param mask:numpy
    :param img_path:
    :return:
    """
    cmap = plt.cm.get_cmap('jet').copy()
    cmap.set_under('black')
    img = img + 10 * (mask - 1)
    plt.imsave(
        os.path.join(img_path),
        img,
        vmin=-1,
        vmax=1,
        cmap=cmap
    )

# --------------------
#   训练准备
# --------------------
def write_config(config, config_path):
    """
    write config file
    """
    with open(config_path, 'w') as jsonfile:
        json.dump(config, jsonfile, indent='\t')


def read_config(config_path):
    """
    load config
    """
    try:
        with open(config_path, 'r') as jsonfile:
            try:
                config = json.load(jsonfile)
            except JSONDecodeError:
                print('Not valid json doc!')
                sys.exit()
    except FileNotFoundError:
        print(f'no config file found at \'{config_path}\'')
        sys.exit()
    return config


def name_folder(
        suffix: Optional[str] = None
):
    naming = time.strftime(f'%Y%m%d_%H%M%S_%A', time.localtime())
    if suffix:
        naming += '_' + suffix
    return naming


# --------------------
#   训练中间过程记录
# --------------------

class AverageMeter(object):
    """
    跟踪记录类，用于统计一组数据的平均值、累加和、数据个数.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LossSaver:
    """

    """
    def __init__(self):
        self.loss_list = []

    def reset(self):
        self.loss_list = []

    def append(self, loss):
        self.loss_list.append(loss)

    def to_np_array(self):
        return np.array(self.loss_list)

    def save_to_file(self, file_path):
        np_loss_list = np.array(self.loss_list)
        np.save(file_path, np_loss_list)


def load_loss_file(file_path):
    data = np.load(file_path)
    return data


# --------------------
#   image transforms
# --------------------

def compose_transforms(d_min=0, d_max=400, crop: Optional[int] = None, flip: bool = False):
    return compose_input_transforms(crop, flip), \
           compose_mask_transforms(crop, flip), \
           compose_target_transforms(d_min, d_max, crop=crop, flip=flip)


# 输入变换
# todo gaps 层的数据分布是否需要统计
def compose_input_transforms(crop: Optional[int] = None, flip: bool = False):
    trans = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(141.01774070236965,), std=(59.57186979412488,)
        ),
        DtypeTransform(),
    ]
    if crop:
        trans.append(transforms.RandomCrop(crop))
    if flip:
        trans.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )
    return transforms.Compose(trans)


# 遮罩变换
def compose_mask_transforms(crop: Optional[int] = None, flip: bool = False):
    trans = [
        transforms.ToTensor(),
        DtypeTransform(),
    ]
    if crop:
        trans.append(transforms.RandomCrop(crop))
    if flip:
        trans.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )
    return transforms.Compose(trans)


# 目标变换
def compose_target_transforms(d_min=0, d_max=400, crop: Optional[int] = None, flip: bool = False):
    trans = [
        transforms.ToTensor(),
        RangeNorm((d_min, d_max), (-1, 1)),
        DtypeTransform(),
    ]
    if crop:
        trans.append(transforms.RandomCrop(crop))
    if flip:
        trans.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )
    return transforms.Compose(trans)


# 目标反变换
def compose_anti_target_transforms(d_min=0, d_max=400):
    trans = [
        RangeNorm((-1, 1), (d_min, d_max)),
    ]
    return transforms.Compose(trans)


# 目标反变换
def compose_target2input_transforms(d_min=0, d_max=400):
    trans = [
        RangeNorm((-1, 1), (d_min, d_max)),
        transforms.Normalize(
            mean=(141.01774070236965,), std=(59.57186979412488,)
        )
    ]
    return transforms.Compose(trans)


# 目标值域范围变换
class RangeNorm:
    """
    input: [di_min, di_max]
    output:[do_min, do_max]
    """
    def __init__(self, din_minmax: tuple, dout_minmax: tuple):
        self.di_min = din_minmax[0]
        self.di_max = din_minmax[1]
        self.do_min = dout_minmax[0]
        self.do_max = dout_minmax[1]

    def __call__(self, img):
        img = (img - self.di_min) / (self.di_max - self.di_min) * (self.do_max - self.do_min) + self.do_min
        return img


class DtypeTransform:
    """
    to torch.float32
    """
    def __call__(self, img):
        return img.to(torch.float32)


# --------------------
#   dataloader
# --------------------
class SimuHeatCollater:
    """
    默认的就可以用
    """
    def __init__(self, *params):
        self.params = params

    def __call__(self, batch_list):
        """在这里重写collate_fn函数"""
        mask = torch.cat([item[1] for item in batch_list])
        distribs = torch.cat(
            [torch.cat(item[0]) for item in batch_list]
        ).transpose(0, 1)
        return distribs, mask

