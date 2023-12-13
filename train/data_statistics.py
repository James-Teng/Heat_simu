import numpy as np

import datasets
import pp

def get_stat(train_dataset, channel):
    """
    Compute mean and std for training data, every sample should have same total pixel number
    :param train_dataset:
    :param channel:
    :return: (mean, std)
    """

    mean = np.zeros(channel)
    square = np.zeros(channel)
    max_d = np.array([-np.inf for i in range(channel)])
    min_d = np.array([np.inf for i in range(channel)])
    for i in range(len(train_dataset)):
        dis, _, _, _, _, _ = train_dataset[i]
        assert isinstance(dis, list), 'variable \'dis\' should be a list'
        dis[0] = dis[0].reshape((channel, -1))
        for d in range(channel):
            data_except_0 = np.delete(dis[0][d, :], np.where(dis[0][d, :] == 0))
            mean[d] += data_except_0.mean()
            square[d] += np.square(data_except_0).mean()
            max_d[d] = np.max([data_except_0.max(), max_d[d]])
            min_d[d] = np.min([data_except_0.min(), min_d[d]])

    mean /= len(train_dataset)
    square /= len(train_dataset)
    std = np.sqrt(square - np.square(mean))

    return list(mean), list(std), list(min_d), list(max_d)


if __name__ == '__main__':

    datasets_dict = {
        time_interval:
            datasets.SimuHeatDataset(
                time_interval=time_interval,
                roots=[
                    pp.abs_path('data/data3_gap/tensor_format_2interval/gap0.1'),
                    pp.abs_path('data/data3_gap/tensor_format_2interval/gap0.2'),
                    pp.abs_path('data/data3_gap/tensor_format_2interval/gap0.3'),
                    pp.abs_path('data/data3_gap/tensor_format_2interval/gap0.4'),
                    pp.abs_path('data/data3_gap/tensor_format_2interval/gap0.5'),
                    pp.abs_path('data/data3_gap/tensor_format_2interval/gap0.6'),
                    pp.abs_path('data/data3_gap/tensor_format_2interval/gap0.7'),
                    pp.abs_path('data/data3_gap/tensor_format_2interval/gap0.8'),
                    pp.abs_path('data/data3_gap/tensor_format_2interval/gap0.9'),
                    pp.abs_path('data/data3_gap/tensor_format_2interval/gap1.0'),
                ],
                gaps=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,],
                supervised_range=0,
                flip=False,
                crop_size=None,
                is_transform=False,
            )
        for time_interval in ['1000.0', '10.0']
    }

    for time_interval, dataset in datasets_dict.items():
        mean, std, min_d, max_d = get_stat(dataset, 1)
        print(f'{time_interval}: mean={mean}, std={std}, min={min_d}, max={max_d}')

