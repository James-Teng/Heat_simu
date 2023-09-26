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

# 输入数据变换,可以加入 crop 和 rotate
def compose_input_transforms():
    trans = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=141.01774070236965, std=59.57186979412488
        ),
    ]
    return transforms.Compose(trans)


# 遮罩变换
def compose_mask_transforms():
    trans = [
        transforms.ToTensor(),
    ]
    return transforms.Compose(trans)


# 目标变换
def compose_target_transforms(d_min=0, d_max=400):
    trans = [
        transforms.ToTensor(),
        TargetRangeNorm(d_min, d_max)
    ]
    return transforms.Compose(trans)


# 目标值域范围变换
class TargetRangeNorm:
    """
    input: [d_min, d_max]
    output:[-1, 1]
    """
    def __init__(self, d_min, d_max):
        self.d_min = d_min
        self.d_max = d_max

    def __call__(self, img):
        img = (img - self.d_min) / (self.d_max - self.d_min) * 2 - 1
        return img


# target to input 变换
def target2input_transforms():
    pass

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


#########################################################
def input_transforms(img):
    """
    非数据集图像做超分的时候用
    """
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean and std
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return trans(img)


def out_transform(img):
    """
    from [-1, 1] to [0, 1]
    """
    img = (img + 1.) / 2.
    return img

