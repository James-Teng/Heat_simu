import os
import numpy as np
import matplotlib.pyplot as plt

path = r'E:\Research\Project\Heat_simu\data\data2_even\tensor_format'
distrib = np.load(os.path.join(path, '102_make_mask.npy'))

plt.figure()
cmap = plt.cm.get_cmap('jet').copy()
img = plt.imshow(distrib, vmin=1, vmax=400, cmap=cmap)
img.cmap.set_under('black')
plt.axis('off')

# 壳体区域
condition1 = np.logical_or(
    np.logical_and(1 < distrib, distrib < 73.5),  # 内壳
    np.logical_and(185 < distrib, distrib < 200),  # 外壳
)

region1 = np.where(condition1, 1, 0)

plt.figure()
img1 = plt.imshow(region1)
plt.axis('off')

# 需要预测的区域
condition2 = np.logical_or(
    distrib == 0,  # 无数据区域
    np.logical_and(185 < distrib, distrib < 200),  # 外壳
)

region2 = np.where(condition2, 0, 1)

plt.figure()
img2 = plt.imshow(region2)
plt.axis('off')
plt.show()

# 保存 region
np.save(os.path.join(path, f'region_casing.npy'), region1)
np.save(os.path.join(path, f'region_supervised.npy'), region2)
