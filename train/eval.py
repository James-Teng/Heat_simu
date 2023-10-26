# todo 前向传播延时
# todo 显示模型的训练配置

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023.10.26 18:32
# @Author  : James.T
# @File    : eval.py

import sys
import os
import argparse
import time
import json
from tqdm import tqdm

import torch.backends.cudnn as cudnn
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
from torchvision.utils import make_grid

import utils
from arch import SimpleArchR
import datasets
import eval_metrics


if __name__ == '__main__':

    checkpoint_path = r'E:\Research\Project\Heat_simu\training_record\20231026_163352_Thursday_test\checkpoint\checkpoint.pth'
    eval_save_path = r'E:\Research\Project\Heat_simu\eval_record'

    # --------------------------------------------------------------------
    #  config
    # --------------------------------------------------------------------

    # path
    folder_name = utils.name_folder()
    record_path = os.path.join(eval_save_path, folder_name)

    # create folders
    os.makedirs(record_path)

    # load config
    large_kernel_size = 9
    small_kernel_size = 3
    n_channels = 32
    n_blocks = 4

    # choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'\nusing {device} device\n')

    cudnn.benchmark = True  # 加速卷积

    # --------------------------------------------------------------------
    #  Initialization
    # --------------------------------------------------------------------

    start_time = time.time()

    # SRResNet
    model = SimpleArchR(
        large_kernel_size=large_kernel_size,
        small_kernel_size=small_kernel_size,
        in_channels=2,
        n_channels=n_channels,
        n_blocks=n_blocks,
    )

    # resume
    if os.path.isfile(checkpoint_path):
        print(f'Checkpoint found, loading...')
        checkpoint = torch.load(checkpoint_path)

        # load model weights
        model.load_state_dict(checkpoint['model'])
    else:
        raise FileNotFoundError(f'No checkpoint found at \'{checkpoint_path}\'')

    # to device
    model = model.to(device)
    model.eval()

    # dataset
    eval_dataset = datasets.DatasetFromFolder(
        roots=[
            r'E:\Research\Project\Heat_simu\data\data2_even\tensor_format\0.1K_0.3gap',
        ],  # 此处暂时不支持命令行配置
        gaps=[
            0.3,
        ],
        supervised_range=1,
        transform_input=utils.compose_input_transforms(),
        transform_region=utils.compose_mask_transforms(),
        transform_target=utils.compose_target_transforms(),
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    # --------------------------------------------------------------------
    #  evaluation
    # --------------------------------------------------------------------

    strtime = time.strftime(f'%Y.%m.%d %H:%M:%S %A', time.localtime())
    print('\n{:-^70}\n'.format(f' eval started at {strtime} '))

    PSNRs = utils.AverageMeter()
    SSIMs = utils.AverageMeter()
    time_costs = utils.AverageMeter()

    cnt = 0
    with torch.no_grad():
        for x, y, mask in tqdm(eval_dataloader, leave=True):

            x = x.to(device)
            y = torch.cat(y, dim=0).to(device)
            mask = mask.to(device)

            predict = model(x)

            y_masked = (y * mask).squeeze()
            predict_masked = (predict * mask).squeeze()
            psnr = eval_metrics.psnr(y_masked, predict_masked, data_range=2)   # 如何去除空白像素点
            ssim = eval_metrics.ssim(y_masked, predict_masked, data_range=2)
            PSNRs.update(psnr, y.shape[0])
            SSIMs.update(ssim, y.shape[0])

            utils.plt_save_image(
                y[0, 0, :, :].cpu().numpy(),
                mask[0, 0, :, :].cpu().numpy(),
                os.path.join(record_path, f'{cnt}_gt.png'),
            )
            utils.plt_save_image(
                predict[0, 0, :, :].cpu().detach().numpy(),
                mask[0, 0, :, :].cpu().numpy(),
                os.path.join(record_path, f'{cnt}_p.png'),
            )
            cnt += 1

    print(f'\nPSNR {PSNRs.avg:.3f}')
    print(f'SSIM {SSIMs.avg:.3f}')

    strtime = time.strftime(f'%Y.%m.%d %H:%M:%S %A', time.localtime())
    print('\n{:-^70}\n'.format(f' eval finished at {strtime} '))
    total_time = time.time() - start_time
    cost_time = time.strftime(f'%H:%M:%S', time.gmtime(total_time))
    print(f'total eval costs {cost_time}')

