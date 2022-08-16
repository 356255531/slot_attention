import numpy as np
from PIL import Image
import tqdm


mat_1 = np.loadtxt('mnist_all_rotation_normalized_float_train_valid.amat')
mat_2 = np.loadtxt('mnist_all_rotation_normalized_float_test.amat')

size = 56
final_size = 32
threshold = 28

for j in tqdm.tqdm(range(100000)):
    N = np.random.randint(3) + 1
    digits = mat_1[np.random.randint(10000, size=N)][:, :-1].reshape(N, 28, 28)
    digits = np.swapaxes(digits, 1, 2)
    img = np.zeros((size, size))
    while 1:
        coor = np.random.randint(0, size - 28, size=2 * N).reshape(N, 2)
        coor_dis = np.sum(np.abs(coor[:, None, :] - coor[None, :, :]), axis=-1)
        if np.sum(np.sum(coor_dis[~np.eye(coor_dis.shape[0],dtype=bool)] < threshold)) == 0:
            break
    for i in range(N):
        img[coor[i, 0]: coor[i, 0] + 28, coor[i, 1]: coor[i, 1] + 28] += digits[i]
    img[img > 1] = 1
    img = (255.0 / img.max().max() * (img - img.min())).astype(np.uint8)
    img = Image.fromarray(img)
    img.thumbnail((final_size, final_size), Image.Resampling.BILINEAR)
    img.save(f'images/train/{j}.png')

for j in tqdm.tqdm(range(5000)):
    N = np.random.randint(3) + 1
    digits = mat_1[np.random.randint(10000, 12000, size=4)][:, :-1].reshape(4, 28, 28)
    digits = np.swapaxes(digits, 1, 2)
    img = np.zeros((size, size))
    while 1:
        coor = np.random.randint(0, size - 28, size=2 * N).reshape(N, 2)
        coor_dis = np.sum(np.abs(coor[:, None, :] - coor[None, :, :]), axis=-1)
        if np.sum(np.sum(coor_dis[~np.eye(coor_dis.shape[0],dtype=bool)] < threshold)) == 0:
            break
    for i in range(N):
        img[coor[i, 0]: coor[i, 0] + 28, coor[i, 1]: coor[i, 1] + 28] += digits[i]
    img[img > 1] = 1
    img = (255.0 / img.max().max() * (img - img.min())).astype(np.uint8)
    img = Image.fromarray(img)
    img.thumbnail((final_size, final_size), Image.Resampling.BILINEAR)
    img.save(f'images/val/{j}.png')

for j in tqdm.tqdm(range(5000)):
    N = np.random.randint(3) + 1
    digits = mat_2[np.random.randint(50000, size=N)][:, :-1].reshape(N, 28, 28)
    digits = np.swapaxes(digits, 1, 2)
    img = np.zeros((size, size))
    while 1:
        coor = np.random.randint(0, size - 28, size=2 * N).reshape(N, 2)
        coor_dis = np.sum(np.abs(coor[:, None, :] - coor[None, :, :]), axis=-1)
        if np.sum(np.sum(coor_dis[~np.eye(coor_dis.shape[0],dtype=bool)] < threshold)) == 0:
            break
    for i in range(N):
        img[coor[i, 0]: coor[i, 0] + 28, coor[i, 1]: coor[i, 1] + 28] += digits[i]
    img[img > 1] = 1
    img = (255.0 / img.max().max() * (img - img.min())).astype(np.uint8)
    img = Image.fromarray(img)
    img.thumbnail((final_size, final_size), Image.Resampling.BILINEAR)
    img.save(f'images/test/{j}.png')
