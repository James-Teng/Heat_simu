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

import torch.backends.cudnn as cudnn
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
from torchvision.utils import make_grid

import utils
from arch import SimpleArchR
import datasets

# bug: 出过一次多线程的问题 DataLoader worker (pid(s) 20940) exited unexpectedly

# 可以在命令行中新建任务配置并开始训练
# 可以指定任务路径来恢复 resume 训练，resume 中需要包含 config

if __name__ == '__main__':

    is_record_iter = False
    train_save_path = r'E:\Research\Project\Heat_simu\training_record'

    # --------------------------------------------------------------------
    #  config
    # --------------------------------------------------------------------

    # arg
    parser = argparse.ArgumentParser(description='train Simuheat')
    parser.add_argument("--brief", "-bf", type=str, default=None, help="brief description")
    parser.add_argument("--train_dataset_root", "-td", type=str, default=None, help="the root of training dataset")
    parser.add_argument("--crop_size", "-cp", type=int, default=None, help="crop size when training")
    parser.add_argument("--supervised_range", "-sr", type=int, default=1, help="supervised steps")

    parser.add_argument("--large_kernel_size", "-lk", type=int, default=9, help="large conv kernel size")
    parser.add_argument("--small_kernel_size", "-sk", type=int, default=3, help="small conv kernel size")
    parser.add_argument("--channels", "-ch", type=int, default=32, help="conv channels")
    parser.add_argument("--blocks", "-bk", type=int, default=2, help="the number of residual blocks")

    parser.add_argument("--epoch", "-ep", type=int, default=100, help="total epochs to train")
    parser.add_argument("--batch_size", "-bs", type=int, default=16, help="batch size")
    parser.add_argument("--lr_initial", "-lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--lr_decay_gamma", "-lr_dg", type=float, default=1e-1, help="learning rate decay gamma")
    parser.add_argument("--lr_milestones", "-ms", type=str, default=None, help="lr milestones eg: 1,2,3")  # 需要一个转换成list的函数

    parser.add_argument("--n_gpu", "-gpu", type=int, default=1, help="number of gpu")
    parser.add_argument("--worker", "-wk", type=int, default=0, help="dataloader worker")

    parser.add_argument("--resume", "-r", type=str, default=None, help="the path of previous weights and configs")
    args = vars(parser.parse_args())

    # todo 恢复 config 文件

    # path
    folder_name = utils.name_folder(args['brief'])
    task_path = os.path.join(train_save_path, folder_name)
    checkpoint_path = os.path.join(task_path, 'checkpoint')
    record_path = os.path.join(task_path, 'record')

    # create folders
    os.makedirs(task_path)
    os.mkdir(checkpoint_path)
    os.mkdir(record_path)

    # display config
    print('\n{:-^52}\n'.format(' TASK CONFIG '))
    print(json.dumps(args, indent='\t'))

    # load config
    dataset_root = args['train_dataset_root']
    crop_size = args['crop_size']
    supervised_range = args['supervised_range']

    large_kernel_size = args['large_kernel_size']
    small_kernel_size = args['small_kernel_size']
    n_channels = args['channels']
    n_blocks = args['blocks']

    total_epochs = args['epoch']
    batch_size = args['batch_size']
    lr = args['lr_initial']

    n_gpu = args['n_gpu']
    worker = args['worker']

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
        n_channels=n_channels,
        n_blocks=n_blocks,
    )

    # todo resume
    # get resume file
    start_epoch = 0
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print(f'Checkpoint found, loading...')
    #         checkpoint = torch.load(args.resume)
    #
    #         # load model weights
    #         start_epoch = checkpoint['epoch'] + 1
    #         model.load_state_dict(checkpoint['model'])
    #
    #     else:
    #         raise FileNotFoundError(f'No checkpoint found at \'{args.resume}\'')
    #
    # else:
    #     print('train from scratch')

    # to device
    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    # optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)  # filter(lambda p: p.requires_grad, model.parameters())

    # load optimizer
    # if args.resume:
    #     optimizer.load_state_dict(checkpoint['optimizer'])

    # n gpu
    is_multi_gpu = torch.cuda.is_available() and n_gpu > 1
    if is_multi_gpu:
        model = nn.DataParallel(model, device_ids=list(range(n_gpu)))  # 之后的项目应该用 nn.DistributedDataParallel

    # dataset
    train_dataset = datasets.DatasetFromFolder(
        root=dataset_root,
        supervised_range=supervised_range,
        transform_input=utils.compose_input_transforms(),
        transform_mask=utils.compose_mask_transforms(),
        transform_target=utils.compose_target_transforms(),
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
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
        for x, y, mask in per_epoch_bar:

            # to device
            x = x.to(device)
            y = torch.cat(y, dim=0).to(device)
            mask = mask.to(device)

            # forward
            predict = model(x*mask)

            # loss
            loss = criterion(predict*mask, y*mask)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # update weights
            optimizer.step()

            # record loss
            if is_record_iter:
                loss_iters_list.append(loss.item())
            loss_epoch.update(loss.item(), mask.shape[0])  # per epoch loss

        # record img change
        utils.plt_save_image(
            y[0, 0, :, :].cpu().numpy(),
            mask[0, 0, :, :].cpu().numpy(),
            os.path.join(record_path, f'epoch_{epoch}_gt.png'),
        )
        utils.plt_save_image(
            predict[0, 0, :, :].cpu().detach().numpy(),
            mask[0, 0, :, :].cpu().numpy(),
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
                'config': args,
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

