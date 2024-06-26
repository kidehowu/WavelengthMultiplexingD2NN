import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt


def random_matrices_generation(N, H, W):
    matrices_data = torch.rand((N, H, W))
    phase_matrices = torch.rand((N, H, W)) * 2 * torch.pi
    matrices_data = matrices_data * torch.exp(1j * phase_matrices)
    return matrices_data


def loader_list_generation(n, n_multiplexing, input_fov, batch_size):
    transform_dimension = input_fov ** 2
    tr_dataloaders = []
    te_dataloaders = []
    train_size = int(0.9 * n)
    test_size = n - train_size

    target_transform_matrices = random_matrices_generation(n_multiplexing, transform_dimension, transform_dimension)
    for i in range(n_multiplexing):
        data_matrices = random_matrices_generation(n, input_fov, input_fov)
        label_matrices = data_matrices.view(data_matrices.size(0), -1, 1)
        label_matrices = torch.matmul(target_transform_matrices[i], label_matrices)
        label_matrices = label_matrices.view(label_matrices.size(0), input_fov, -1)
        dataset = TensorDataset(data_matrices, label_matrices)
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        tr_dataloaders.append(train_loader)
        te_dataloaders.append(test_loader)
    return target_transform_matrices, tr_dataloaders, te_dataloaders


def plot_cosine_similarity_matrix(data):
    data = data.view(len(data), -1)
    data = data / data.norm(dim=1, keepdim=True)
    data = torch.abs(torch.mm(data, torch.conj(data.t())).real)

    fig = plt.figure()
    plt.imshow(data, cmap=plt.cm.Blues)
    thresh = data.max() / 2
    for i in range(len(data)):
        for j in range(len(data)):
            info = data[j, i]
            plt.text(i, j, format(info, ".2f"),
                     verticalalignment='center',
                     horizontalalignment='center',
                     color="white" if info > thresh else "black")
    plt.title("Cosine Similarity Matrix")
    plt.show()
    return fig


def mse_loss():
    def mse(output, target):
        x = output - target
        x = (x * torch.conj(x)).real.mean()
        return x

    return mse
