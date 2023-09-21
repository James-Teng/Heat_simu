# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023.9.20 14:27
# @Author  : James.T
# @File    : all_txt_to_tensor.py

import os
import matplotlib.pyplot as plt
import numpy as np


# 文件不大的话，可以整个文件一起放入内存，

# to do：
# 每一帧的时间信息没有处理
# 解决数据中 nan 的问题，暂时换成 0


data_path = r'E:\Research\Project\Heat_simu\data\data2_even\txt_format\0.1K_0.1gap.txt'
# data_path = r'E:\Research\Project\Heat_simu\data\data2_even\txt_format\0.1K_0.3gap.txt'
output_path = r'E:\Research\Project\Heat_simu\data\data2_even\tensor_format'

max_T = 400  # 摄氏度
min_T = 0

if __name__ == '__main__':

    with open(data_path, 'r', encoding='utf-8') as f:

        # 将文件开头的信息丢掉
        print('\n{:-^52}\n'.format(' DATA INFO '))
        for column in range(8):
            line = f.readline().strip('\n')  # 读取一行，并去掉换行符
            print(line)

        # 处理时间信息，未完成,以下三行丢掉了时间信息
        line = f.readline().strip('\n')
        line_split = line.split()  # 按照空格进行分割
        print(line[1:3])

        # 读取采样点分布与一帧
        print('\n{:-^52}\n'.format(' Processing all data '))
        points = []
        data = []
        mask = []
        line = f.readline().strip('\n')
        run_once = True

        while line:

            # 分割数据
            line_split = line.split()  # 按照空格进行分割

            # 读取采样点坐标信息
            points.append([float(x) for x in line_split[:2]])  # 前两位为坐标，格式为字符串，需要转浮点数

            # 读取某一位置所有时间的数据，并替换 NaN
            # 此处简单赋值为 0，需要验证合理
            data.append([float(0) if x == 'NaN' else float(x) for x in line_split[2:]])  # 替换所有 NaN 并将数据转换为浮点数

            # 生成对应遮罩
            mask.append(float(0) if line_split[2] == 'NaN' else float(1))

            # 统计帧总数
            if run_once:
                print(f'total frame: {len(line_split) - 2}')
                run_once = False

            # 读取下一行
            line = f.readline().strip('\n')

        # 统计单帧张量大小
        column = 0
        row = 1
        r_former = points[0]
        for r in points:
            if r[0] >= r_former[0]:
                column += 1
                r_former = r
            else:
                column = 1
                row += 1
                r_former = points[0]
        print(f'tensor size: row - {row}, column - {column}')  # 横向(r) 130 纵向(z) 260
        print(f'resolution: ({points[1][0] - points[0][0]}, {points[column][1] - points[0][1]}) mm')

        # 转置矩阵并重构
        all_distrib = np.array(data).transpose((1, 0)).reshape((-1, row, column))
        mask = np.array(mask).reshape((row, column))

        # 保存训练数据
        dir_name, _ = os.path.splitext(os.path.basename(data_path))
        output_dir = os.path.join(output_path, dir_name)
        os.makedirs(output_dir)

        np.save(os.path.join(output_dir, f'mask_{dir_name}.npy'), mask)  # 保存list，读取需要手动转换 list()
        plt.imsave(os.path.join(output_dir, f'mask_{dir_name}.png'), mask)

        for i in range(all_distrib.shape[0]):
            np.save(
                os.path.join(output_dir, f'{i}.npy'),
                all_distrib[i],
            )
            plt.imsave(
                os.path.join(output_dir, f'{i}.png'),
                all_distrib[i],
                vmin=min_T,
                vmax=max_T,
                cmap='jet'
            )

        # 保存基本信息，未完成


