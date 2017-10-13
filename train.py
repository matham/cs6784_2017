#!/usr/bin/env python3

import argparse
from PIL import Image
import torch
import numpy as np
from random import shuffle

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset

import os
import sys
import math

import shutil

import densenet
import make_graph


class SplitCifarDataSet(Dataset):

    dataset = None

    data = None

    labels = None

    def __init__(self, dataset, classes):
        self.dataset = dataset

        if dataset.train:
            orig_data = dataset.train_data
            orig_labels = dataset.train_labels
        else:
            orig_data = dataset.test_data
            orig_labels = dataset.test_labels
        orig_labels = np.array(orig_labels, dtype=np.int32)

        classes = set(classes)
        select = np.zeros((len(orig_labels), ), dtype=bool)
        for i, label in enumerate(orig_labels):
            select[i] = label in classes

        self.data = orig_data[select, :, :, :]
        self.labels = orig_labels[select]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.dataset.transform is not None:
            img = self.dataset.transform(img)

        if self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=64)
    parser.add_argument('--nEpochs', type=int, default=175)
    parser.add_argument('--trans', action='store_true')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--dataRoot')
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--cifar', type=int, default=10,
                        choices=(10, 100))
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or 'work/densenet.base'

    cifar10 = args.cifar == 10

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    net = densenet.DenseNet(growthRate=12, depth=100, reduction=0.5,
                            bottleneck=True, nClasses=(10 if cifar10 else 100))

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    if args.cuda:
        net = net.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1,
                            momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

    if cifar10:
        normMean = [0.53129727, 0.52593911, 0.52069134]
        normStd = [0.28938246, 0.28505746, 0.27971658]
    else:
        normMean =  [0.5423671, 0.53410053, 0.52827841]
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

    if args.trans:
        res = run_transfer(args, optimizer, net, trainTransform, testTransform)
        run_transfer_dset_b(args, *res)
    else:
        run(args, optimizer, net, trainTransform, testTransform)

def run(args, optimizer, net, trainTransform, testTransform):

    cifar10 = args.cifar == 10
    download = not args.dataRoot
    data_root = args.dataRoot or 'cifar'

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    cifar_cls = dset.CIFAR10 if cifar10 else dset.CIFAR100

    train_set = cifar_cls(
        root=data_root, train=True, download=download, transform=trainTransform)
    trainLoader = DataLoader(
        train_set, batch_size=args.batchSz, shuffle=True, **kwargs)

    test_set = cifar_cls(
        root=data_root, train=False,
        download=download, transform=testTransform)
    testLoader = DataLoader(
        test_set, batch_size=args.batchSz, shuffle=False, **kwargs)

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')

    best_error = 100
    for epoch in range(1, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, net, trainLoader, optimizer, trainF)
        err = test(args, epoch, net, testLoader, optimizer, testF)

        if err < best_error:
            best_error = err
            print('New best error {}'.format(err))
            torch.save(net.state_dict(), os.path.join(args.save, 'model_cifar{}.t7'.format(args.cifar)))

        os.system('./plot.py {} &'.format(args.save))

    trainF.close()
    testF.close()

def run_transfer(args, optimizer, net, trainTransform, testTransform, resume=False):
    cifar10 = args.cifar == 10
    N = args.cifar
    download = not args.dataRoot
    data_root = args.dataRoot or 'cifar'

    if resume:
        with open(os.path.join(args.save, 'class_shuffled'), 'r') as fh:
            classes = list(map(int, fh.read().split(',')))
    else:
        classes = list(range(N))
        shuffle(classes)
        with open(os.path.join(args.save, 'class_shuffled'), 'w') as fh:
            fh.write(','.join(map(str, classes)))

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    cifar_cls = dset.CIFAR10 if cifar10 else dset.CIFAR100

    train_set = cifar_cls(
        root=data_root, train=True, download=download, transform=trainTransform)
    tran1 = SplitCifarDataSet(train_set, classes[:N // 2])
    tran2 = SplitCifarDataSet(train_set, classes[N // 2:])

    test_set = cifar_cls(
        root=data_root, train=False,
        download=download, transform=testTransform)
    test1 = SplitCifarDataSet(test_set, classes[:N // 2])
    test2 = SplitCifarDataSet(test_set, classes[N // 2:])

    trainLoader = DataLoader(
        tran1, batch_size=args.batchSz, shuffle=True, **kwargs)
    testLoader = DataLoader(
        test1, batch_size=args.batchSz, shuffle=False, **kwargs)

    trainF = open(os.path.join(args.save, 'train1.csv'), 'w')
    testF = open(os.path.join(args.save, 'test1.csv'), 'w')

    best_error = 100
    best_state = net.state_dict()
    if not resume:
        for epoch in range(1, args.nEpochs + 1):
            adjust_opt(args.opt, optimizer, epoch)
            train(args, epoch, net, trainLoader, optimizer, trainF)
            err = test(args, epoch, net, testLoader, optimizer, testF)

            if err < best_error:
                best_error = err
                best_state = net.state_dict()
                print('New best error {}'.format(err))
                torch.save(best_state, os.path.join(args.save, 'model_cifar{}_base.t7'.format(args.cifar)))

            # os.system('./plot.py {} &'.format(args.save))

    trainF.close()
    testF.close()
    return tran2, test2, os.path.join(args.save, 'model_cifar{}_base.t7'.format(args.cifar))


def run_transfer_dset_b(args, tran2, test2, filename):
    cifar10 = args.cifar == 10

    net = densenet.DenseNet(growthRate=12, depth=100, reduction=0.5,
                            bottleneck=True, nClasses=(10 if cifar10 else 100))
    if args.cuda:
        net = net.cuda()

    net.load_state_dict(torch.load(filename))
    net.reset_last_layer()

    params = list(net.parameters())
    fc_params = list(net.fc.parameters())
    base_params = [p for p in params if not [fc_p for fc_p in fc_params if fc_p is p]]
    param_vals = [
        {'params': fc_params, 'lr': 1e-1},
        {'params': base_params, 'lr': 1e-2}
    ]

    optimizer = optim.SGD(param_vals, lr=1e-1, momentum=0.9, weight_decay=1e-4)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    trainLoader = DataLoader(
        tran2, batch_size=args.batchSz, shuffle=True, **kwargs)
    testLoader = DataLoader(
        test2, batch_size=args.batchSz, shuffle=False, **kwargs)

    trainF = open(os.path.join(args.save, 'train2.csv'), 'w')
    testF = open(os.path.join(args.save, 'test2.csv'), 'w')

    best_error = 100
    best_state = net.state_dict()
    for epoch in range(1, 100 + 1):
        adjust_opt_transfer(args.opt, optimizer, epoch)
        train(args, epoch, net, trainLoader, optimizer, trainF)
        err = test(args, epoch, net, testLoader, optimizer, testF)

        if err < best_error:
            best_error = err
            best_state = net.state_dict()
            print('New best error {}'.format(err))
            torch.save(best_state, os.path.join(args.save, 'model_cifar{}_trans.t7'.format(args.cifar)))

        # os.system('./plot.py {} &'.format(args.save))

    trainF.close()
    testF.close()

def train(args, epoch, net, trainLoader, optimizer, trainF):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.data[0], err))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
        trainF.flush()

def test(args, epoch, net, testLoader, optimizer, testF):
    net.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(testLoader) # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 100.*incorrect/nTotal
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, incorrect, nTotal, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()
    return err

def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch == 1: lr = 1e-1
        elif epoch == 126: lr = 1e-2
        elif epoch == 151: lr = 1e-3
        else: return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def adjust_opt_transfer(optAlg, optimizer, epoch):
    fc, base = optimizer.param_groups
    if epoch == 51:
        fc['lr'] = base['lr'] = 1e-2
    elif epoch == 76:
        fc['lr'] = base['lr'] = 1e-3

if __name__=='__main__':
    main()
