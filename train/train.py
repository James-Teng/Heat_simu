# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023.9.26 18:32
# @Author  : James.T
# @File    : train.py

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

import utils
from arch import SimpleArchR
import datasets

import training_manage

# bug: 出过一次多线程的问题 DataLoader worker (pid(s) 20940) exited unexpectedly

if __name__ == '__main__':

    is_record_iter = False
    train_save_path = r'./training_record'

    # --------------------------------------------------------------------
    #  config
    # --------------------------------------------------------------------

    # arg
    parser = argparse.ArgumentParser(description='train Simuheat')
    parser.add_argument("--brief", "-bf", type=str, default=None, help="brief description")

    # 数据集
    parser.add_argument("--crop_size", "-cp", type=int, default=None, help="crop size when training")
    parser.add_argument("--supervised_range", "-sr", type=int, default=1, help="supervised steps")
    parser.add_argument("--flip", type=bool, default=True, help="flip")
    parser.add_argument(
        "--time_intervals", "-ti", nargs='+', type=str,
        default=[
            # '1000.0',
            '10.0'
        ],
        help="time intervals"
    )
    parser.add_argument(
        "--data_roots", "-dr", nargs='+', type=str,
        default=[
            r'./data/data3_gap/tensor_format_2interval/gap0.1',  # 数据所在的文件夹
            r'./data/data3_gap/tensor_format_2interval/gap0.2',
            r'./data/data3_gap/tensor_format_2interval/gap0.3',
            r'./data/data3_gap/tensor_format_2interval/gap0.4',
            r'./data/data3_gap/tensor_format_2interval/gap0.5',
            r'./data/data3_gap/tensor_format_2interval/gap0.6',
            r'./data/data3_gap/tensor_format_2interval/gap0.7',
            r'./data/data3_gap/tensor_format_2interval/gap0.8',
            r'./data/data3_gap/tensor_format_2interval/gap0.9',
            r'./data/data3_gap/tensor_format_2interval/gap1.0',
        ],
        help="where data is",
    )
    parser.add_argument("--enable_gaps", type=bool, default=True, help="involving gaps feature")  # 没有实现在数据集和模型上
    parser.add_argument(
        "--gaps", "-gap", nargs='+', type=float,
        default=[
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ],
        help="shell gaps",
    )

    # 模型
    parser.add_argument("--large_kernel_size", "-lk", type=int, default=9, help="large conv kernel size")
    parser.add_argument("--small_kernel_size", "-sk", type=int, default=3, help="small conv kernel size")
    parser.add_argument("--in_channels", "-ic", type=int, default=2, help="input channels")  # 没有使用
    parser.add_argument("--channels", "-ch", type=int, default=32, help="conv channels")
    parser.add_argument("--blocks", "-bk", type=int, default=4, help="the number of residual blocks")
    parser.add_argument("--initial_weight", "-iw", type=str, default=None, help="path of initial weights")

    # 训练
    parser.add_argument("--epoch", "-ep", type=int, default=100, help="total epochs to train")
    parser.add_argument("--batch_size", "-bs", type=int, default=24, help="batch size")
    parser.add_argument("--lr_initial", "-lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--lr_decay_gamma", "-lr_dg", type=float, default=1, help="learning rate decay gamma")
    parser.add_argument("--lr_milestones", "-ms", nargs='+', type=int, default=[], help="lr milestones eg: 1 2 3")

    parser.add_argument("--n_gpu", "-gpu", type=int, default=1, help="number of gpu")
    parser.add_argument("--worker", "-wk", type=int, default=0, help="dataloader worker")

    parser.add_argument("--resume", "-r", type=str, default=None, help="the path of previous training")
    args = vars(parser.parse_args())

    # # resume
    resume = args['resume']
    del args['resume']
    if resume:
        task_path = resume
        checkpoint_path = os.path.join(task_path, 'checkpoint')
        record_path = os.path.join(task_path, 'record')
        config_path = os.path.join(task_path, 'config.json')
        args = training_manage.read_config(config_path)

    # # new training
    else:
        # new path
        folder_name = utils.name_folder(args['brief'])
        task_path = os.path.join(train_save_path, folder_name)
        checkpoint_path = os.path.join(task_path, 'checkpoint')
        record_path = os.path.join(task_path, 'record')
        config_path = os.path.join(task_path, 'config.json')

        # create folders
        os.makedirs(task_path)
        os.mkdir(checkpoint_path)
        os.mkdir(record_path)

    # display config
    print('\n{:-^52}\n'.format(' TASK CONFIG '))
    print(json.dumps(args, indent='\t'))

    # todo 考虑是否有必要存在变量里
    # here to load config
    time_intervals = args['time_intervals']
    roots = args['data_roots']
    gaps = args['gaps']
    crop_size = args['crop_size']
    flip = args['flip']
    supervised_range = args['supervised_range']

    large_kernel_size = args['large_kernel_size']
    small_kernel_size = args['small_kernel_size']
    in_channels = args['in_channels']
    n_channels = args['channels']
    n_blocks = args['blocks']
    initial_weight = args['initial_weight']

    total_epochs = args['epoch']
    batch_size = args['batch_size']
    lr = args['lr_initial']
    lr_decay_gamma = args['lr_decay_gamma']
    lr_milestone = args['lr_milestones']

    n_gpu = args['n_gpu']
    worker = args['worker']

    # 存储配置
    if not resume:
        training_manage.write_config(args, config_path)

    # --------------------------------------------------------------------
    #  Initialization
    # --------------------------------------------------------------------

    # choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'\nusing {device} device\n')
    cudnn.benchmark = True  # 加速卷积

    start_time = time.time()

    # SRResNet
    model = SimpleArchR(
        large_kernel_size=large_kernel_size,
        small_kernel_size=small_kernel_size,
        in_channels=in_channels,
        n_channels=n_channels,
        n_blocks=n_blocks,
    )

    # get resume file
    start_epoch = 0
    if resume:
        if os.path.isfile(os.path.join(checkpoint_path, f'checkpoint.pth')):  # 判断需要恢复的任务checkpoint是否存在
            print(f'Checkpoint found, loading...')
            checkpoint = torch.load(os.path.join(checkpoint_path, f'checkpoint.pth'))

            # load model weights
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])

        else:
            raise FileNotFoundError(f'No checkpoint found at \'{checkpoint_path}\'')

    else:
        print('train from scratch')

    # to device
    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    # optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)  # 可以过滤需要梯度的权重

    # load optimizer
    if resume:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # n gpu
    is_multi_gpu = torch.cuda.is_available() and n_gpu > 1
    if is_multi_gpu:
        model = nn.DataParallel(model, device_ids=list(range(n_gpu)))  # 之后的项目应该用 nn.DistributedDataParallel

    # datasets
    datasets_dict = datasets.SimuHeatDataset(
        time_intervals=time_intervals,
        roots=roots,
        gaps=gaps,
        supervised_range=supervised_range,
        flip=flip,
        crop_size=crop_size,
    )

    train_dataloader = torch.utils.data.DataLoader(
        datasets_dict[time_intervals[0]],
        batch_size=batch_size,
        shuffle=True,
        num_workers=worker,
        pin_memory=True,
    )

    # --------------------------------------------------------------------
    #  training
    # --------------------------------------------------------------------

    strtime = time.strftime(f'%Y.%m.%d %H:%M:%S %A', time.localtime())
    print('\n{:-^70}\n'.format(f' training started at {strtime} '))

    loss_epochs_list = utils.LossSaver()  # record loss per epoch
    if is_record_iter:
        loss_iters_list = utils.LossSaver()  # record loss per iter
    writer = SummaryWriter(record_path)  # tensorboard

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', record_path])
    url = tb.launch()

    total_bar = tqdm(range(start_epoch, total_epochs), desc='[Total Progress]')
    for epoch in total_bar:

        model.train()

        loss_epoch = utils.AverageMeter()
        per_epoch_bar = tqdm(train_dataloader, leave=False)
        for x, y, casing, supervised, data, outer in per_epoch_bar:

            # to device
            x_casing = datasets.cat_input(x, casing)  # 叠加输入 # todo 输入的过程搬到模型中进行，模型封装一层，可以更换 backbone
            x_casing = x_casing.to(device)
            y = torch.cat(y, dim=0).to(device)
            supervised = supervised.to(device)

            # forward
            predict = model(x_casing)

            # loss
            loss = criterion(predict * supervised, y * supervised)  # todo 增加迭代监督
            # loss = criterion(predict, y * mask)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # update weights
            optimizer.step()

            # record loss
            if is_record_iter:
                loss_iters_list.append(loss.item())
            loss_epoch.update(loss.item(), supervised.shape[0])  # per epoch loss

        # record img change
        utils.plt_save_image(
            y[0, 0, :, :].cpu().numpy(),
            supervised[0, 0, :, :].cpu().numpy(),
            os.path.join(record_path, f'epoch_{epoch}_gt.png'),
        )
        utils.plt_save_image(
            predict[0, 0, :, :].cpu().detach().numpy(),
            supervised[0, 0, :, :].cpu().numpy(),
            os.path.join(record_path, f'epoch_{epoch}_p.png'),
        )
        # writer.add_image(f'{task_id}/epoch_{epoch}_lr', lr_gird)

        # record loss
        loss_epochs_list.append(loss_epoch.val)
        writer.add_scalar(f'{folder_name}/MSE_Loss', loss_epoch.val, epoch)

        # todo eval
        # todo save best model
        # save model
        torch.save(
            {
                'epoch': epoch,
                'model': model.module.state_dict() if is_multi_gpu else model.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
            os.path.join(checkpoint_path, f'checkpoint.pth'),
        )

    # save loss file
    loss_epochs_list.save_to_file(os.path.join(record_path, f'epoch_loss_{start_epoch}_{total_epochs-1}.npy'))
    if is_record_iter:
        loss_iters_list.save_to_file(os.path.join(record_path, f'iter_loss_{start_epoch}_{total_epochs-1}.npy'))

    writer.close()

    strtime = time.strftime(f'%Y.%m.%d %H:%M:%S %A', time.localtime())
    print('\n{:-^70}\n'.format(f' training finished at {strtime} '))
    total_time = time.time() - start_time
    cost_time = time.strftime(f'%H:%M:%S', time.gmtime(total_time))
    print(f'total training costs {cost_time}')

