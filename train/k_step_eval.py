# todo 保存参数来源，在什么数据集上测试的
# todo 记录 psnr 最大最小

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023.10.26 18:32
# @Author  : James.T
# @File    : eval.py

import _init_cwd  # change cwd

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

# todo 适配 train config
if __name__ == '__main__':

    eval_save_path = r'./eval_record'

    parser = argparse.ArgumentParser(description='k step eval')
    parser.add_argument("-k", type=int, default=10, help="k steps forward")
    parser.add_argument("--checkpoint", '-c', type=str, default=None, help="weight path")
    args = vars(parser.parse_args())
    checkpoint_path = args['checkpoint']
    k = args['k']

    # 数据集配置
    time_intervals = [  # 指定时间间隔
        '1000.0',
        # '10.0',
        # '0.1',
    ]
    roots = [
            r'./data/data3_gap/tensor_format_2interval/gap0.1',  # 数据所在的文件夹
            # r'./data/data3_gap/tensor_format_2interval/gap0.2',
            # r'./data/data3_gap/tensor_format_2interval/gap0.3',
            # r'./data/data3_gap/tensor_format_2interval/gap0.4',
            # r'./data/data3_gap/tensor_format_2interval/gap0.5',
            # r'./data/data3_gap/tensor_format_2interval/gap0.6',
            # r'./data/data3_gap/tensor_format_2interval/gap0.7',
            # r'./data/data3_gap/tensor_format_2interval/gap0.8',
            # r'./data/data3_gap/tensor_format_2interval/gap0.9',
            # r'./data/data3_gap/tensor_format_2interval/gap1.0',
    ]
    gaps = [
        0.1,
        # 0.2,
        # 0.3,
        # 0.4,
        # 0.5,
        # 0.6,
        # 0.7,
        # 0.8,
        # 0.9,
        # 1.0,
    ]
    # --------------------------------------------------------------------
    #  config
    # --------------------------------------------------------------------

    # path
    folder_name = utils.name_folder()
    record_path = os.path.join(eval_save_path, folder_name)
    record_info_path = os.path.join(record_path, 'info.txt')

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
    datasets_dict = datasets.SimuHeatDataset(
        time_intervals=time_intervals,
        roots=roots,
        gaps=gaps,
        supervised_range=k,
        flip=False,
        crop_size=None
    )
    eval_dataloader = torch.utils.data.DataLoader(
        datasets_dict[time_intervals[0]],
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    # record files
    comment = time.strftime(f'%Y-%m-%d_%H-%M-%S', time.localtime())
    with open(record_info_path, 'x') as f:
        f.write(f'-- {k} steps eval --\n')
        f.write(f'model path: {checkpoint_path}\n')
        f.write(f'datasets: {roots}\n')
        f.write(f'time interval: {time_intervals}\n')
        f.write(f'eval time: {comment}\n')

    # --------------------------------------------------------------------
    #  evaluation
    # --------------------------------------------------------------------

    strtime = time.strftime(f'%Y.%m.%d %H:%M:%S %A', time.localtime())
    print('\n{:-^70}\n'.format(f' eval started at {strtime} '))

    PSNRs = utils.AverageMeter()
    SSIMs = utils.AverageMeter()
    time_costs = utils.AverageMeter()

    t2i_trans = utils.compose_target2input_transforms()

    cnt = 0
    with torch.no_grad():
        for x, y, casing, supervised, data, outer in tqdm(eval_dataloader, leave=True):

            x = x.to(device)
            y = torch.stack(y, dim=0).to(device)

            casing = casing.to(device)
            supervised = supervised.to(device)
            data = data.to(device)
            outer = outer.to(device)

            interval = datasets.cat_input(x, casing)
            # k steps forward
            for ii in range(k):
                interval = model(interval)
                if ii == k-1:
                    break
                # mask非生成区域，配置正确的外壳温度，变换为输入，mask无数据区域，叠加区域
                interval = interval * supervised + y[ii] * outer
                interval = t2i_trans(interval) * data
                interval = datasets.cat_input(interval, casing)

            y_masked = (y[-1] * supervised).squeeze()
            predict_masked = (interval * supervised).squeeze()
            psnr = eval_metrics.psnr(y_masked, predict_masked, data_range=2)  # 如何去除空白像素点
            ssim = eval_metrics.ssim(y_masked, predict_masked, data_range=2)
            PSNRs.update(psnr, y.shape[0])
            SSIMs.update(ssim, y.shape[0])

            utils.plt_save_image(
                # range batch channel h w
                y[-1, 0, 0, :, :].cpu().numpy(),
                supervised[0, 0, :, :].cpu().numpy(),
                os.path.join(record_path, f'{cnt}_gt.png'),
            )
            utils.plt_save_image(
                interval[0, 0, :, :].cpu().detach().numpy(),
                supervised[0, 0, :, :].cpu().numpy(),
                os.path.join(record_path, f'{cnt+k}_p.png'),
            )
            cnt += 1

    print(f'\nPSNR {PSNRs.avg:.3f}')
    print(f'SSIM {SSIMs.avg:.3f}')

    with open(record_info_path, 'a') as f:
        # f.write(f'\nevaluate on {dataset_name}\n')
        f.write(f'PSNR {PSNRs.avg:.3f}\n')
        f.write(f'SSIM {SSIMs.avg:.3f}\n')
        # f.write(f'costs {t_cost_per:.3f} per image\n')

    strtime = time.strftime(f'%Y.%m.%d %H:%M:%S %A', time.localtime())
    print('\n{:-^70}\n'.format(f' eval finished at {strtime} '))
    total_time = time.time() - start_time
    cost_time = time.strftime(f'%H:%M:%S', time.gmtime(total_time))
    print(f'total eval costs {cost_time}')

