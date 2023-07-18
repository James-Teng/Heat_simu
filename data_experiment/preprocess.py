import os
import matplotlib.pyplot as plt

data_path = r'E:\Research\Project\Heat_simu\data\data1\no_initial_dam\gap0.5_0.1Kpermin-1.txt'

if __name__ == '__main__':

    # 绘制数据点分布
    with open(data_path, 'r', encoding='utf-8') as f:

        # 将文件开头的信息丢掉
        print('\n{:-^52}\n'.format(' DATA INFO '))
        for i in range(8):
            line = f.readline().strip('\n')  # 读取一行，并去掉换行符
            print(line)

        # 处理时间信息，未完成
        line = f.readline().strip('\n')  # 读取一行，并去掉换行符
        line_split = line.split()  # 按照空格进行分割
        print(line[1:3])

        # 读取点分布
        print('\n{:-^52}\n'.format(' POINTS '))
        points = []
        line = f.readline().strip('\n')
        while line:
            line_split = line.split()  # 按照空格进行分割
            # print(line[:2])
            points.append([float(x) for x in line_split[:2]])  # 字符串转数字
            line = f.readline().strip('\n')

        # 绘制散点分布图
        plt.figure(figsize=(500, 500))
        plt.scatter(
            [point[0] for point in points],
            [point[1] for point in points],
            s=5
        )
        ax = plt.gca()
        ax.set_aspect(1)
        plt.show()
        # print(points)

