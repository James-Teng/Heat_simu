# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023.9.26 18:32
# @Author  : James.T
# @File    : train.py

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

import utils
import arch
import datasets
import training_manage
import pp


# bug: 出过一次多线程的问题 DataLoader worker (pid(s) 20940) exited unexpectedly

if __name__ == '__main__':

    is_record_iter = False
    train_save_path = pp.abs_path('training_record')

    # --------------------------------------------------------------------
    #  config
    # --------------------------------------------------------------------

    # arg
    parser = argparse.ArgumentParser(description='train Simuheat')
    parser.add_argument("--brief", "-bf", type=str, default=None, help="brief description")

    # dataset setting
    parser.add_argument("--crop_size", "-cp", type=int, default=None, help="crop size when training")
    parser.add_argument("--k_steps_supervised", "-k", type=int, default=1, help="supervised steps")
    parser.add_argument("--flip", type=bool, default=True, help="flip")
    parser.add_argument(  # todo 修改数据集关于时间间隔的加载
        "--time_intervals", "-ti", nargs='+', type=str,  # choices=['1000.0', '10.0', '0.1'],
        default=[
            '1000.0',
            # '10.0'
            # '0.1',
        ],
        help="time intervals"
    )
    parser.add_argument(
        "--data_roots", "-dr", nargs='+', type=str,
        default=[
            pp.abs_path('data/data3_gap/tensor_format_2interval/gap0.1'),  # 数据所在的文件夹
            pp.abs_path('data/data3_gap/tensor_format_2interval/gap0.2'),
            pp.abs_path('data/data3_gap/tensor_format_2interval/gap0.3'),
            pp.abs_path('data/data3_gap/tensor_format_2interval/gap0.4'),
            pp.abs_path('data/data3_gap/tensor_format_2interval/gap0.5'),
            pp.abs_path('data/data3_gap/tensor_format_2interval/gap0.6'),
            # pp.abs_path('data/data3_gap/tensor_format_2interval/gap0.7'),
            pp.abs_path('data/data3_gap/tensor_format_2interval/gap0.8'),
            pp.abs_path('data/data3_gap/tensor_format_2interval/gap0.9'),
            pp.abs_path('data/data3_gap/tensor_format_2interval/gap1.0'),
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
            # 0.7,
            0.8,
            0.9,
            1.0,
        ],
        help="shell gaps",
    )

    # model structure
    parser.add_argument("--model_type", "-m", type=str, default='IterativeFramework', help="model type")
    parser.add_argument("--large_kernel_size", "-lk", type=int, default=9, help="large conv kernel size")
    parser.add_argument("--small_kernel_size", "-sk", type=int, default=3, help="small conv kernel size")
    parser.add_argument("--in_channels", "-ic", type=int, default=2, help="input channels")  # 没有使用
    parser.add_argument("--n_channels", "-ch", type=int, default=32, help="conv channels")
    parser.add_argument("--blocks", "-bk", type=int, default=4, help="the number of residual blocks")
    parser.add_argument("--bn", action="store_true", help="batch normalization")
    parser.add_argument("--initial_weight", "-iw", type=str, default=None, help="path of initial weights")

    # training strategy
    parser.add_argument("--epochs", "-ep", type=int, default=100, help="total epochs to train")
    parser.add_argument("--batch_size", "-bs", type=int, default=24, help="batch size")
    parser.add_argument("--lr_initial", "-lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--lr_decay_gamma", "-lr_dg", type=float, default=1, help="learning rate decay gamma")
    parser.add_argument("--lr_milestones", "-ms", nargs='+', type=int, default=[], help="lr milestones eg: 1 2 3")

    parser.add_argument("--n_gpu", "-gpu", type=int, default=1, help="number of gpu")
    parser.add_argument("--worker", "-wk", type=int, default=10, help="dataloader worker")

    parser.add_argument("--resume", "-r", type=str, default=None, help="the path of previous training")
    parser.add_argument("--debug", "-d", action="store_true", help="debug mode")
    args = vars(parser.parse_args())

    logging.basicConfig(level=logging.DEBUG if args['debug'] else logging.WARNING)

    # todo 增加命令行参数检测

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

    # save config
    if not resume:
        training_manage.write_config(args, config_path)

    # --------------------------------------------------------------------
    #  Initialization
    # --------------------------------------------------------------------

    # choose device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cudnn.benchmark = True  # 加速卷积
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f'\nusing {device} device\n')

    start_time = time.time()

    # define model
    model = arch.model_factory(args)

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
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args['lr_initial'])  # 可以过滤需要梯度的权重

    # load optimizer
    if resume:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # n gpu
    is_multi_gpu = torch.cuda.is_available() and args['n_gpu'] > 1
    if is_multi_gpu:
        model = nn.DataParallel(model, device_ids=list(range(args['n_gpu'])))  # 之后的项目应该用 nn.DistributedDataParallel

    # datasets
    datasets_dict = datasets.SimuHeatDataset(
        time_intervals=args['time_intervals'],
        roots=args['data_roots'],
        gaps=args['gaps'],
        supervised_range=args['k_steps_supervised'],
        flip=args['flip'],
        crop_size=args['crop_size'],
    )

    train_dataloader = torch.utils.data.DataLoader(
        datasets_dict[args['time_intervals'][0]],
        batch_size=args['batch_size'],
        shuffle=True,
        num_workers=args['worker'],
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

    total_bar = tqdm(range(start_epoch, args['epochs']), desc='[Total Progress]')
    for epoch in total_bar:

        model.train()

        loss_epoch = utils.AverageMeter()
        per_epoch_bar = tqdm(train_dataloader, leave=False)
        for x, y, casing, supervised, data, outer in per_epoch_bar:

            # to device
            x, y = x.to(device), y.to(device)
            casing = casing.to(device)
            supervised = supervised.to(device)
            # data = data.to(device)
            outer = outer.to(device)
            # model.set_region(casing, supervised, data, outer)

            # forward
            predict = model(x, casing, supervised, data, outer)
            logging.debug(f"predict: {predict.shape}, y s: {(y * supervised.unsqueeze(1)).shape}")

            # loss
            loss = criterion(predict, y * supervised.unsqueeze(1))

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
            y[0, -1, 0, :, :].cpu().numpy(),
            supervised[0, 0, :, :].cpu().numpy(),
            os.path.join(record_path, f'epoch_{epoch}_gt.png'),
        )
        utils.plt_save_image(
            predict[0, -1, 0, :, :].detach().cpu().numpy(),
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
    loss_epochs_list.save_to_file(os.path.join(record_path, f"epoch_loss_{start_epoch}_{args['epochs']-1}.npy"))
    if is_record_iter:
        loss_iters_list.save_to_file(os.path.join(record_path, f"iter_loss_{start_epoch}_{args['epochs']-1}.npy"))

    writer.close()

    strtime = time.strftime(f'%Y.%m.%d %H:%M:%S %A', time.localtime())
    print('\n{:-^70}\n'.format(f' training finished at {strtime} '))
    total_time = time.time() - start_time
    cost_time = time.strftime(f'%H:%M:%S', time.gmtime(total_time))
    print(f'total training costs {cost_time}')

