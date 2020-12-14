import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import torch
import cv2


def load_data_mnist(batch_size):
    mnist_test = torchvision.datasets.MNIST(root='./MNIST', train=False, download=True,
                                            transform=transforms.ToTensor())
    mnist_train = torchvision.datasets.MNIST(root='./MNIST', train=True, download=True,
                                             transform=transforms.ToTensor())
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 0
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter


def print_mnist(data_loader_train):
    images, labels = next(iter(data_loader_train))
    img = torchvision.utils.make_grid(images)
    img = img.numpy().transpose(1, 2, 0)
    std = [0.5, 0.5, 0.5]
    mean = [0.5, 0.5, 0.5]
    img = img * std + mean
    print(labels)
    cv2.imshow('win', img)
    cv2.waitKey(0)

