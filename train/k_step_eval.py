# todo 保存参数来源，在什么数据集上测试的
# todo 记录 psnr 最大最小

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023.10.26 18:32
# @Author  : James.T
# @File    : eval.py

import pp  # change cwd

import sys
import os
import argparse
import time
import json
from tqdm import tqdm
import logging

import torch.backends.cudnn as cudnn
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
from torchvision.utils import make_grid

import utils
import arch
import datasets
import eval_metrics
import training_manage

# todo 适配 train config
if __name__ == '__main__':

    eval_save_path = pp.abs_path('eval_record')

    parser = argparse.ArgumentParser(description='k step eval')
    parser.add_argument("-k", type=int, default=10, help="k steps forward", required=True)
    parser.add_argument("--path", '-p', type=str, default=None, help="task path", required=True)
    parser.add_argument("--debug", '-d', action='store_true', help="debug mode")
    args = vars(parser.parse_args())
    logging.basicConfig(level=logging.DEBUG if args['debug'] else logging.WARNING)
    task_path = args['path']
    k = args['k']

    # --------------------------------------------------------------------
    #  config
    # --------------------------------------------------------------------

    # load config
    config = training_manage.read_config(
        os.path.join(task_path, 'config.json')
    )

    data_roots = [
        pp.abs_path('data/data3_gap/tensor_format_2interval/gap0.1'),  # 数据所在的文件夹
        # pp.abs_path('data/data3_gap/tensor_format_2interval/gap0.2'),
        # pp.abs_path('data/data3_gap/tensor_format_2interval/gap0.3'),
        # pp.abs_path('data/data3_gap/tensor_format_2interval/gap0.4'),
        # pp.abs_path('data/data3_gap/tensor_format_2interval/gap0.5'),
        # pp.abs_path('data/data3_gap/tensor_format_2interval/gap0.6'),
        # pp.abs_path('data/data3_gap/tensor_format_2interval/gap0.7'),
        # pp.abs_path('data/data3_gap/tensor_format_2interval/gap0.8'),
        # pp.abs_path('data/data3_gap/tensor_format_2interval/gap0.9'),
        # pp.abs_path('data/data3_gap/tensor_format_2interval/gap1.0'),
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

    # choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'\nusing {device} device\n')

    cudnn.benchmark = True  # 加速卷积

    # path
    folder_name = utils.name_folder()
    record_path = os.path.join(eval_save_path, folder_name)
    record_info_path = os.path.join(record_path, 'info.txt')

    # create folders
    os.makedirs(record_path)

    # --------------------------------------------------------------------
    #  Initialization
    # --------------------------------------------------------------------

    start_time = time.time()

    # SRResNet
    model = arch.NaiveRNNFramework(
        extractor=arch.SimpleExtractor(
            in_channels=config['in_channels'],
            out_channels=config['n_channels'],
            kernel_size=config['large_kernel_size'],
        ),
        backbone=arch.SimpleBackbone(
            n_blocks=config['blocks'],
            n_channels=config['n_channels'],
            kernel_size=config['small_kernel_size'],
        ),
        regressor=arch.SimpleRegressor(
            kernel_size=config['large_kernel_size'],
            in_channels=config['n_channels'],
            out_channels=1,
        ),
        out2intrans=utils.compose_target2input_transforms()
    )

    # resume
    checkpoint_path = os.path.join(task_path, 'checkpoint/checkpoint.pth')
    if os.path.isfile(checkpoint_path):
        print(f'Checkpoint found, loading...')
        checkpoint = torch.load(checkpoint_path)

        # load model weights
        model.load_state_dict(checkpoint['model'])  # todo 统计加载率
    else:
        raise FileNotFoundError(f'No checkpoint found at \'{checkpoint_path}\'')

    # to device
    model = model.to(device)
    model.eval()

    # dataset
    datasets_dict = datasets.SimuHeatDataset(
        time_intervals=config['time_intervals'],
        roots=data_roots,
        gaps=gaps,
        supervised_range=k,
        flip=False,
        crop_size=None
    )
    eval_dataloader = torch.utils.data.DataLoader(
        datasets_dict[config['time_intervals'][0]],
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    # record files
    comment = time.strftime(f'%Y-%m-%d_%H-%M-%S', time.localtime())
    with open(record_info_path, 'x') as f:
        f.write(f'-- {k} steps eval --\n')
        f.write(f'model path: {checkpoint_path}\n')
        f.write(f'datasets: {data_roots}\n')
        f.write(f"time interval: {config['time_intervals']}\n")
        f.write(f'eval time: {comment}\n')

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
        for x, y, casing, supervised, data, outer in tqdm(eval_dataloader, leave=True):

            x = x.to(device)
            y = y.to(device)
            casing = casing.to(device)
            supervised = supervised.to(device)
            data = data.to(device)
            outer = outer.to(device)

            # k steps forward
            predicts = model(x, casing, supervised, data, outer)

            # # record from step 1 to k
            # for p, target in predicts[0, :, :, :, :], y[0, :, :, :, :]:
            #     psnr = eval_metrics.psnr(target, p, data_range=2)
            #     ssim = eval_metrics.ssim(target, p, data_range=2)

            logging.debug(f'predicts shape: {predicts.shape}')
            logging.debug(f'predicts[0, -1, :, :, :] shape: {predicts[0, -1, :, :, :].shape}')
            logging.debug(f'y shape: {y.shape}')

            final_predict = predicts[0, -1, :, :, :].squeeze()
            final_target = (y[:, -1, :, :, :] * supervised).squeeze()

            logging.debug(f'final_predict shape: {final_predict.shape}')
            logging.debug(f'final_predict: {final_predict[40:60, :]}')
            logging.debug(f'final_target shape: {final_target.shape}')

            psnr = eval_metrics.psnr(final_predict, final_target, data_range=2)  # todo 去掉无数据点
            ssim = eval_metrics.ssim(final_predict, final_target, data_range=2)
            logging.debug(f'psnr: {psnr}')
            logging.debug(f'ssim: {ssim}')

            PSNRs.update(psnr, 1)
            SSIMs.update(ssim, 1)

            utils.plt_save_image(
                # range batch channel h w
                final_target.cpu().numpy(),
                supervised[0, 0, :, :].cpu().numpy(),
                os.path.join(record_path, f'{cnt}_gt.png'),
            )
            utils.plt_save_image(
                final_predict.cpu().numpy(),
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

