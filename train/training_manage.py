# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023.3.8 15:35
# @Author  : James.T
# @File    : task_manager.py

import argparse
import time
import os
import json
from json.decoder import JSONDecodeError
import sys
from typing import Optional

config_template = {
    'create_time': 'None',
    'description': 'None',

    'train_dataset_config': {
        'train_dataset_root': '',  # what dataset to use
        'crop_size': 96,  # crop
    },
    'model_config': {
        'large_kernel_size': 9,  # 第一层卷积和最后一层卷积的核大小
        'small_kernel_size': 3,  # 中间层卷积的核大小
        'n_channels': 32,  # 中间层通道数
        'n_blocks': 2,  # 残差模块数量
    },
    'hyper_params': {
        'total_epochs': None,
        'batch_size': None,
        'lr_initial': None,
        'lr_decay_gamma': None,
        'lr_milestones': [],
    },
    'others': {
        'n_gpu': 1,
        'worker': 4,
    },
    'resume': None
}
#
#
# # 暂时没用
# def new_training_record(
#         task_path: str,
#         checkpoint_path: str,
#         record_path: str,
#         config: Optional[dict] = None,
# ):
#     config_path = os.path.join(task_path, 'config.json')
#     # create path
#     os.makedirs(task_path)
#     os.mkdir(checkpoint_path)
#     os.mkdir(record_path)
#     # config
#     if not config:
#         config = config_template
#     write_config(config, config_path)


# config
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

