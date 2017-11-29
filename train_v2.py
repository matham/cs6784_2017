#!/usr/bin/env python3

import argparse
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
from datasets import CIFAR100
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader

import os
import sys
import math
import numpy as np
from random import shuffle

import shutil

import densenet_v1 as densenet
from train import SplitCifarDataSet
data_dir = '/root/Desktop/data'
results_dir = 'results_v2'
batch = 64
bin_weight = 0.4
baseline_epocs = 175
ft_epochs = 100
classes_a = np.array([
    31, 52, 92, 56, 54, 40, 50, 23, 88, 35, 87, 85, 25, 45, 68, 51, 2,
    27, 9, 84, 59, 42, 72, 77, 73, 80, 95, 13, 65, 79, 64, 97, 47, 66,
    53, 0, 94, 98, 90, 5, 1, 82, 38, 16, 74, 48, 71, 63, 29, 67],
    dtype=np.int64)
classes_a_bin = classes_a[:25]
use_best_state = True


def main():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    normMean = [0.5423671, 0.53410053, 0.52827841]
    normStd = [0.30129549, 0.29579896, 0.29065931]
    normTransform = transforms.Normalize(normMean, normStd)

    trainTransform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normTransform
    ])
    testTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True}

    trainLoaderA = DataLoader(
        CIFAR100(root=data_dir, group='A', train=True, download=False,
                 transform=trainTransform, classes_a=classes_a,
                 classes_a_bin=classes_a_bin),
        batch_size=batch, shuffle=True, **kwargs)
    testLoaderA = DataLoader(
        CIFAR100(root=data_dir, group='A', train=False, download=False,
                 transform=testTransform, classes_a=classes_a,
                 classes_a_bin=classes_a_bin),
        batch_size=batch, shuffle=False, **kwargs)

    trainLoaderB = DataLoader(
        CIFAR100(root=data_dir, group='B', train=True, download=False,
                 transform=trainTransform, classes_a=classes_a,
                 classes_a_bin=classes_a_bin),
        batch_size=batch, shuffle=True, **kwargs)
    testLoaderB = DataLoader(
        CIFAR100(root=data_dir, group='B', train=False, download=False,
                 transform=testTransform, classes_a=classes_a,
                 classes_a_bin=classes_a_bin),
        batch_size=batch, shuffle=False, **kwargs)

    def get_net(transfer=False, binary=False):
        net = densenet.DenseNet(growthRate=12, depth=100, reduction=0.5,
                                bottleneck=True, nClasses=100, binary=binary)
        if transfer:
            ft_params, reset_params = net.split_parmeters()
            params = [
                {'params': reset_params, 'lr': 1e-1},
                {'params': ft_params, 'lr': 1e-2}
            ]
        else:
            params = net.parameters()
        optimizer = optim.SGD(params, lr=1e-1,
                            momentum=0.9, weight_decay=1e-4)
        net = net.cuda()
        print(net)
        return net, optimizer

    net, optimizer = get_net(transfer=False)
    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))

    best_state = run_base(net, trainLoaderA, testLoaderA, optimizer, binary=False)
    net.reset_last_layer()
    state = best_state if use_best_state else net.state_dict()

    net, optimizer = get_net(transfer=True)
    net.load_state_dict(state)
    net.reset_last_layer()

    run_transfer(net, trainLoaderB, testLoaderB, optimizer, binary=False)

    net, optimizer = get_net(binary=True, transfer=False)
    best_state = run_base(net, trainLoaderA, testLoaderA, optimizer, binary=True)
    net.reset_last_layer()
    state = best_state if use_best_state else net.state_dict()

    net, optimizer = get_net(transfer=True)
    net.load_state_dict(state)
    net.reset_last_layer()

    run_transfer(net, trainLoaderB, testLoaderB, optimizer, binary=True)


def run_base(net, trainLoader, testLoader, optimizer, binary):
    trainF = open(os.path.join(
        results_dir, 'trainA{}.csv'.format('_bin' if binary else '')), 'w')
    testF = open(os.path.join(
        results_dir, 'testA{}.csv'.format('_bin' if binary else '')), 'w')

    transfer = False
    best_loss = 1000
    best_state = net.state_dict()
    for epoch in range(1, baseline_epocs + 1):
        adjust_opt(optimizer, epoch)
        train(epoch, net, trainLoader, optimizer, trainF, binary, transfer)
        new_loss = test(epoch, net, testLoader, testF, binary, transfer)
        if new_loss < best_loss:
            best_loss = new_loss
            best_state = net.state_dict()
            print('new best results')

    trainF.close()
    testF.close()
    return best_state


def run_transfer(net, trainLoader, testLoader, optimizer, binary):
    trainF = open(os.path.join(
        results_dir, 'trainB{}.csv'.format('_bin' if binary else '')), 'w')
    testF = open(os.path.join(
        results_dir, 'testB{}.csv'.format('_bin' if binary else '')), 'w')

    transfer = True
    for epoch in range(1, ft_epochs + 1):
        adjust_opt_transfer(optimizer, epoch)
        train(epoch, net, trainLoader, optimizer, trainF, False, transfer)
        test(epoch, net, testLoader, testF, False, transfer)

    trainF.close()
    testF.close()


def train(epoch, net, trainLoader, optimizer, trainF, binary, transfer):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        if binary and not transfer:  # A binary
            data, target, target_bin = data.cuda(), target[0].cuda(), target[1].cuda()
            data, target, target_bin = Variable(data), Variable(target), Variable(target_bin)
            optimizer.zero_grad()
            fc_out, bin_out = net(data)
            loss_fc = F.nll_loss(fc_out, target)
            loss_bin = F.nll_loss(bin_out, target_bin)
            loss = bin_weight * loss_bin + (1. - bin_weight) * loss_fc
        elif not transfer:  # A not binary
            #print(type(target), target[0].size(), target[1].size())
            data, target = data.cuda(), target[0].cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            fc_out, bin_out = net(data)
            #print(type(fc_out), fc_out.size(), type(target), target.size())
            #print(target)
            loss = F.nll_loss(fc_out, target)
        else:  # B
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            fc_out, bin_out = net(data)
            loss = F.nll_loss(fc_out, target)

        loss.backward()
        optimizer.step()

        nProcessed += len(data)
        pred = fc_out.data.max(1)[1]  # get the index of the max log-probability

        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.data[0], err))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
        trainF.flush()


def test(epoch, net, testLoader, testF, binary, transfer):
    net.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader:
        if binary and not transfer:
            data, target, target_bin = data.cuda(), target[0].cuda(), target[1].cuda()
            data, target, target_bin = Variable(data, volatile=True), Variable(target), Variable(target_bin)
            fc_out, bin_out = net(data)
            # print(data[0])
            # raise Exception
            test_loss += F.nll_loss(fc_out, target).data[0]
        elif not transfer:
            data, target = data.cuda(), target[0].cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            fc_out, bin_out = net(data)
            test_loss += F.nll_loss(fc_out, target).data[0]
        else:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            fc_out, bin_out = net(data)
            test_loss += F.nll_loss(fc_out, target).data[0]

        pred = fc_out.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss /= len(testLoader) # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 100.*incorrect/nTotal
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, incorrect, nTotal, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()
    return test_loss


def adjust_opt(optimizer, epoch):
    if epoch == 1:
        lr = 1e-1
    elif epoch == 126:
        lr = 1e-2
    elif epoch == 151:
        lr = 1e-3
    else:
        return

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_opt_transfer(optimizer, epoch):
    fc, base = optimizer.param_groups
    if epoch == 51:
        fc['lr'] = base['lr'] = 1e-2
    elif epoch == 76:
        fc['lr'] = base['lr'] = 1e-3


if __name__=='__main__':
    main()