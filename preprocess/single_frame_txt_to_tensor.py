import os
import matplotlib.pyplot as plt
import numpy as np


# to do：解决数据中 nan 的问题，暂时换成0

def convert_frame(data, index):
    """
    读取文件，并转换指定的一帧
    :param data: 处理成
    :param index:
    :return:
    """
    pass


data_path = r'E:\Research\Project\Heat_simu\data\data2_even\txt_format\0.1K_0.1gap.txt'
output_path = r'E:\Research\Project\Heat_simu\data\data2_even\tensor_format'

frame = 130

if __name__ == '__main__':

    with open(data_path, 'r', encoding='utf-8') as f:

        # 将文件开头的信息丢掉
        print('\n{:-^52}\n'.format(' DATA INFO '))
        for column in range(8):
            line = f.readline().strip('\n')  # 读取一行，并去掉换行符
            print(line)

        # 处理时间信息，未完成,以下三行丢掉了时间信息
        line = f.readline().strip('\n')  # 读取一行，并去掉换行符
        line_split = line.split()  # 按照空格进行分割
        print(line[1:3])

        # 读取点分布
        print('\n{:-^52}\n'.format(' One Frame '))
        points = []
        data = []
        line = f.readline().strip('\n')
        while line:
            line_split = line.split()  # 按照空格进行分割
            points.append([float(x) for x in line_split[:2]])  # 读取坐标信息，字符串转数字
            # 将一帧的数据从字符串转换为数字
            if line_split[frame+2] != 'NaN':
                data.append(float(line_split[frame+2]))  # 读取第一帧的数据，统计最大最小，验证将nan变成0是否合理，之后可能要考虑一下边界的问题
            else:
                data.append(float(0))  # 此处简单赋值为 0，需要验证合理
            # 此处还需要生成一个mask
            line = f.readline().strip('\n')

        # 统计矩阵大小
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
        print(f'row:{row}, column:{column}')  # 横向(r) 130 纵向(z) 260

        # 转换一帧
        # frame = np.zeros((row, column))
        # for y in range(row):
        #     for x in range(column):
        #         frame[y][x] = data[column * y + x]
        distrib = np.array(data).reshape(row, column)

        # visualization
        print(f'the {frame}th frame')
        plt.figure()
        plt.imshow(distrib)
        plt.axis('off')
        plt.show()


