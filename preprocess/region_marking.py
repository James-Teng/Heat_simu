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

# 有数据的区域
condition3 = distrib == 0
region3 = np.where(condition3, 0, 1)
plt.figure()
img3 = plt.imshow(region3)
plt.axis('off')

# 外壳体区域
condition4 = np.logical_and(185 < distrib, distrib < 200)  # 外壳
region4 = np.where(condition4, 1, 0)
plt.figure()
img4 = plt.imshow(region4)
plt.axis('off')
plt.show()


# 保存 region
plt.imsave(os.path.join(path, f'region_casing.png'), region1)
np.save(os.path.join(path, f'region_casing.npy'), region1)

plt.imsave(os.path.join(path, f'region_supervised.png'), region2)
np.save(os.path.join(path, f'region_supervised.npy'), region2)

plt.imsave(os.path.join(path, f'region_data.png'), region3)
np.save(os.path.join(path, f'region_data.npy'), region3)

plt.imsave(os.path.join(path, f'region_outer.png'), region4)
np.save(os.path.join(path, f'region_outer.npy'), region4)
