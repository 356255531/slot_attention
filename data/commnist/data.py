import numpy as np
from PIL import Image
import tqdm
import torch
import kornia
import random as rd


train_data = np.loadtxt('mnist_all_rotation_normalized_float_train_valid.amat')
train_data = train_data[train_data[:, -1] != 0]
train_data = train_data[train_data[:, -1] != 1]
train_data = train_data[train_data[:, -1] != 6]
train_data = train_data[train_data[:, -1] != 8]
train_data = train_data[train_data[:, -1] != 9]
train_data = train_data[:, :-1]
train_data = np.swapaxes(train_data.reshape(-1, 28, 28), 1, 2)
train_data = torch.from_numpy(train_data)

test_data = np.loadtxt('mnist_all_rotation_normalized_float_test.amat')
test_data = test_data[test_data[:, -1] != 0]
test_data = test_data[test_data[:, -1] != 1]
test_data = test_data[test_data[:, -1] != 6]
test_data = test_data[test_data[:, -1] != 8]
test_data = test_data[test_data[:, -1] != 9]
test_data = test_data[:, :-1]
test_data = np.swapaxes(test_data.reshape(-1, 28, 28), 1, 2)
test_data = torch.from_numpy(test_data)

size = 96
threshold = 48

for j in tqdm.tqdm(range(100000)):
    N = np.random.randint(3) + 1
    digits = train_data[torch.randint(int(train_data.shape[0] * 0.8), size=(N,))]
    digits = kornia.geometry.rotate(digits.unsqueeze(1), torch.randint(360, size=(N,)).type(digits.dtype), padding_mode="border")
    img = torch.zeros((size, size))
    while 1:
        coor = torch.randint(size - 28, size=(2 * N,)).reshape(N, 2)
        coor_ = coor.unsqueeze(0).repeat_interleave(repeats=N, dim=0)
        coor_T = torch.transpose(coor_, 0, 1)
        coor_ = coor_.reshape(-1, 2)
        coor_T = coor_T.reshape(-1, 2)
        if N < 2 or torch.sum(torch.sum(torch.nn.PairwiseDistance(p=1)(coor_, coor_T) < threshold)) == N:
            break
    for i in range(N):
        delta_h = torch.cat([coor[:, 0], torch.tensor([size])], dim=-1) - coor[i, 0]
        delta_h = delta_h[delta_h > 0]
        delta_h = torch.min(delta_h)
        h_ratio = delta_h / 28
        h_scale = rd.random() * (h_ratio - 1) + 1

        delta_w = torch.cat([coor[:, 1], torch.tensor([size])], dim=-1) - coor[i, 1]
        delta_w = delta_w[delta_w > 0]
        delta_w = torch.min(delta_w)
        w_ratio = delta_w / 28
        w_scale = rd.random() * (w_ratio - 1) + 1

        digit = kornia.geometry.rescale(digits[i].unsqueeze(0), (h_scale, w_scale)).squeeze(0).squeeze(0)
        h, w = digit.shape

        img[coor[i, 0]: coor[i, 0] + h, coor[i, 1]: coor[i, 1] + w] += digit
    img[img > 1] = 1
    img = (255.0 / img.max().max() * (img - img.min())).type(torch.uint8).numpy()
    img = Image.fromarray(img)
    img.save(f'images/train/{j}.png')

for j in tqdm.tqdm(range(5000)):
    N = np.random.randint(3) + 1
    digits = train_data[torch.randint(int(train_data.shape[0] * 0.8), train_data.shape[0], size=(N,))]
    digits = kornia.geometry.rotate(digits.unsqueeze(1), torch.randint(360, size=(N,)).type(digits.dtype), padding_mode="border")
    img = torch.zeros((size, size))
    while 1:
        coor = torch.randint(size - 28, size=(2 * N,)).reshape(N, 2)
        coor_ = coor.unsqueeze(0).repeat_interleave(repeats=N, dim=0)
        coor_T = torch.transpose(coor_, 0, 1)
        coor_ = coor_.reshape(-1, 2)
        coor_T = coor_T.reshape(-1, 2)
        if N < 2 or torch.sum(torch.sum(torch.nn.PairwiseDistance(p=1)(coor_, coor_T) < threshold)) == N:
            break
    for i in range(N):
        delta_h = torch.cat([coor[:, 0], torch.tensor([size])], dim=-1) - coor[i, 0]
        delta_h = delta_h[delta_h > 0]
        delta_h = torch.min(delta_h)
        h_ratio = delta_h / 28
        h_scale = rd.random() * (h_ratio - 1) + 1

        delta_w = torch.cat([coor[:, 1], torch.tensor([size])], dim=-1) - coor[i, 1]
        delta_w = delta_w[delta_w > 0]
        delta_w = torch.min(delta_w)
        w_ratio = delta_w / 28
        w_scale = rd.random() * (w_ratio - 1) + 1

        digit = kornia.geometry.rescale(digits[i].unsqueeze(0), (h_scale, w_scale)).squeeze(0).squeeze(0)
        h, w = digit.shape

        img[coor[i, 0]: coor[i, 0] + h, coor[i, 1]: coor[i, 1] + w] += digit
    img[img > 1] = 1
    img = (255.0 / img.max().max() * (img - img.min())).type(torch.uint8).numpy()
    img = Image.fromarray(img)
    img.save(f'images/val/{j}.png')

for j in tqdm.tqdm(range(5000)):
    N = np.random.randint(3) + 1
    digits = test_data[np.random.randint(test_data.shape[0], size=(N,))]
    digits = kornia.geometry.rotate(digits.unsqueeze(1), torch.randint(360, size=(N,)).type(digits.dtype), padding_mode="border")
    img = torch.zeros((size, size))
    while 1:
        coor = torch.randint(size - 28, size=(2 * N,)).reshape(N, 2)
        coor_ = coor.unsqueeze(0).repeat_interleave(repeats=N, dim=0)
        coor_T = torch.transpose(coor_, 0, 1)
        coor_ = coor_.reshape(-1, 2)
        coor_T = coor_T.reshape(-1, 2)
        if N < 2 or torch.sum(torch.sum(torch.nn.PairwiseDistance(p=1)(coor_, coor_T) < threshold)) == N:
            break
    for i in range(N):
        delta_h = torch.cat([coor[:, 0], torch.tensor([size])], dim=-1) - coor[i, 0]
        delta_h = delta_h[delta_h > 0]
        delta_h = torch.min(delta_h)
        h_ratio = delta_h / 28
        h_scale = rd.random() * (h_ratio - 1) + 1

        delta_w = torch.cat([coor[:, 1], torch.tensor([size])], dim=-1) - coor[i, 1]
        delta_w = delta_w[delta_w > 0]
        delta_w = torch.min(delta_w)
        w_ratio = delta_w / 28
        w_scale = rd.random() * (w_ratio - 1) + 1

        digit = kornia.geometry.rescale(digits[i].unsqueeze(0), (h_scale, w_scale)).squeeze(0).squeeze(0)
        h, w = digit.shape

        img[coor[i, 0]: coor[i, 0] + h, coor[i, 1]: coor[i, 1] + w] += digit
    img[img > 1] = 1
    img = (255.0 / img.max().max() * (img - img.min())).type(torch.uint8).numpy()
    img = Image.fromarray(img)
    img.save(f'images/test/{j}.png')
