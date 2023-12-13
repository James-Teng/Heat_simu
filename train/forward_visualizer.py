import pp  # change cwd

import os
import argparse
import logging

import torch.backends.cudnn as cudnn
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import utils
import arch
import datasets
import training_manage

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

output_path = pp.abs_path('output')

def save(p, gt, mask, cnt):

    logging.debug(f'p: {p.shape}')
    logging.debug(f'gt: {gt.shape}')
    logging.debug(f'mask: {mask.shape}')

    fig, axs = plt.subplots(ncols=2, squeeze=False)
    fig.suptitle(f'Step {cnt}')
    cmap = plt.cm.get_cmap('jet').copy()
    cmap.set_under('black')

    p = p + 10 * (mask - 1)
    gt = gt + 10 * (mask - 1)

    axs[0, 0].imshow(p, vmin=-1, vmax=1, cmap=cmap)
    axs[0, 0].set_title(f'predict', y=-0.1)
    axs[0, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    axs[0, 1].imshow(gt, vmin=-1, vmax=1, cmap=cmap)
    axs[0, 1].set_title(f'ground truth', y=-0.1)
    axs[0, 1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    fig.savefig(os.path.join(output_path, f'{cnt}_.png'))
    plt.close(fig)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='k step eval')
    parser.add_argument("-k", type=int, default=10, help="k steps forward", required=True)
    parser.add_argument("--path", '-p', type=str, default=None, help="task path", required=True)
    parser.add_argument("--debug", '-d', action='store_true', help="debug mode")
    args = vars(parser.parse_args())
    logging.basicConfig(level=logging.DEBUG if args['debug'] else logging.WARNING)
    task_path = args['path']
    k = args['k']

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

    # --------------------------------------------------------------------
    #  Initialization
    # --------------------------------------------------------------------

    # SRResNet
    model = arch.model_factory(config)

    # resume
    checkpoint_path = os.path.join(task_path, 'checkpoint/checkpoint.pth')
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

    # --------------------------------------------------------------------
    #  forward
    # -------------------------------------------------------------------
    with torch.no_grad():

        x, y, casing, supervised, data, outer = next(iter(eval_dataloader))

        x = x.to(device)
        y = y.to(device)
        casing = casing.to(device)
        supervised = supervised.to(device)
        data = data.to(device)
        outer = outer.to(device)

        # k steps forward
        model.enable_interval_output()
        predicts = model(x, casing, supervised, data, outer)

        cnt = 0
        for p, gt in zip(predicts.squeeze(0), y.squeeze(0)):
            logging.debug(f'p: {p.shape}')
            logging.debug(f'gt: {gt.shape}')
            save(p[0].cpu().numpy(), gt[0].cpu().numpy(), supervised.squeeze(0)[0].cpu().numpy(), cnt)
            cnt += 1

