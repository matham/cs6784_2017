#!/usr/bin/env python3

import argparse
from PIL import Image
import torch
import numpy as np
import random
import time
from random import shuffle
import os

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from densenet_vision import DenseNet as DenseNetVision
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset

import os
import sys
import math

import shutil

import densenet
from attic.label_cifar import unnatural_labels, natural_labels
from densenet_efficient import DenseNetEfficient
import make_graph

imagenet100 = {
    5, 6, 7, 15, 27, 50, 55, 73, 76, 80, 96, 104, 105, 106, 107, 130, 135, 152, 179, 185, 188, 208, 248,
    252, 264, 278, 314, 317, 330, 338, 355, 360, 371, 389, 397, 402, 404, 409, 413, 416, 433, 434, 456,
    471, 478, 481, 490, 491, 519, 530, 534, 535, 546, 555, 558, 566, 579, 581, 600, 609, 628, 634, 652,
    654, 659, 672, 681, 686, 688, 690, 693, 700, 708, 723, 746, 753, 769, 775, 802, 808, 814, 816, 837,
    858, 867, 875, 877, 881, 882, 919, 934, 938, 941, 955, 961, 971, 976, 986, 987, 996}


class ImageFolderSubset(dset.folder.ImageFolder):

    original_idx = []

    def __init__(self, included_classes, *largs, **kwargs):
        super(ImageFolderSubset, self).__init__(*largs, **kwargs)
        self.original_idx = [
            i for i, (_, cls) in enumerate(self.imgs) if cls in included_classes]

    def __getitem__(self, index):
        return super(ImageFolderSubset, self).__getitem__(self.original_idx[index])

    def __len__(self):
        return len(self.original_idx)



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
    parser.add_argument('--transBlocks', action='store_true')
    parser.add_argument('--nTransFTBlockLayersStep', type=int, default=2)
    parser.add_argument('--transFTBlock', type=int, default=0)
    parser.add_argument('--transNatSplit', action='store_true')
    parser.add_argument('--transSplit', type=int, default=50)
    parser.add_argument('--binClasses', type=int, default=0)
    parser.add_argument('--binWeight', type=float, default=.67)
    parser.add_argument('--imagenet', action='store_true')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--dataRoot')
    parser.add_argument('--classes')
    parser.add_argument('--save')
    parser.add_argument('--preTrainedModel')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--cifar', type=int, default=10,
                        choices=(10, 100))
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or 'work/densenet.base'

    cifar10 = args.cifar == 10
    use_imagenet = args.imagenet

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    if use_imagenet:
        net = DenseNetVision(
            growth_rate=32, block_config=[6, 12, 24, 16],
            num_classes=1000, num_init_features=64,
            n_binary_class=args.binClasses, binary_only=args.binWeight == 1.)
    else:
        net = densenet.DenseNet(
            growthRate=12, depth=100, reduction=0.5,
            bottleneck=True, nClasses=(10 if cifar10 else 100),
            n_binary_class=args.binClasses, binary_only=args.binWeight == 1.
        )

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

    if use_imagenet:
        trainTransform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        testTransform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
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

    print(net)
    if args.trans:
        res = run_transfer(args, optimizer, net, trainTransform, testTransform)
        if args.transBlocks:
            run_transfer_dset_b(args, [], *res)
            run_transfer_dset_b(args, [1], *res)
            run_transfer_dset_b(args, [1, 2], *res)

        if args.transFTBlock:
            block = args.transFTBlock
            blocks = list(range(1, block + 1))
            for i in range(0, 16, args.nTransFTBlockLayersStep):
                blocks[-1] = (block, i + 1)
                run_transfer_dset_b(args, blocks, *res)
        else:
            run_transfer_dset_b(args, 'all', *res)
    else:
        run(args, optimizer, net, trainTransform, testTransform)


def run(args, optimizer, net, trainTransform, testTransform):
    data_root = args.dataRoot or 'cifar'
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    if args.imagenet:
        train_set = ImageFolderSubset(
            included_classes=imagenet100, root=os.path.join(data_root, 'train'),
            transform=trainTransform)
        test_set = ImageFolderSubset(
            included_classes=imagenet100, root=os.path.join(data_root, 'val'),
            transform=testTransform)
    else:
        cifar10 = args.cifar == 10
        download = not args.dataRoot
        cifar_cls = dset.CIFAR10 if cifar10 else dset.CIFAR100
        train_set = cifar_cls(
            root=data_root, train=True, download=download, transform=trainTransform)
        test_set = cifar_cls(
            root=data_root, train=False,
            download=download, transform=testTransform)

    trainLoader = DataLoader(
        train_set, batch_size=args.batchSz, shuffle=True, **kwargs)
    testLoader = DataLoader(
        test_set, batch_size=args.batchSz, shuffle=False, **kwargs)

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')

    best_error = 100
    ts0 = time.perf_counter()
    for epoch in range(1, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, net, trainLoader, optimizer, trainF, [])
        err = test(args, epoch, net, testLoader, optimizer, testF, [])

        if err < best_error:
            best_error = err
            print('New best error {}'.format(err))
            torch.save(net.state_dict(), os.path.join(args.save, 'model_cifar{}.t7'.format(args.cifar)))

        os.system('./plot.py {} &'.format(args.save))

    trainF.close()
    testF.close()
    print('Done in {:.2f}s'.format(time.perf_counter() - ts0))


def run_transfer(args, optimizer, net, trainTransform, testTransform):
    cifar10 = args.cifar == 10
    N = args.cifar if not args.imagenet else 100
    download = not args.dataRoot
    data_root = args.dataRoot or 'cifar'

    if args.transNatSplit:
        set1, set2 = natural_labels, unnatural_labels
    else:
        if args.classes:
            with open(args.classes, 'r') as fh:
                classes = list(map(int, fh.read().split(',')))
        else:
            classes = list(imagenet100) if args.imagenet else list(range(N))
            shuffle(classes)
            with open(os.path.join(args.save, 'class_shuffled'), 'w') as fh:
                fh.write(','.join(map(str, classes)))
        set1, set2 = classes[:args.transSplit], classes[args.transSplit:]

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    if args.imagenet:
        train1 = ImageFolderSubset(
            included_classes=set1, root=os.path.join(data_root, 'train'),
            transform=trainTransform)
        test1 = ImageFolderSubset(
            included_classes=set1, root=os.path.join(data_root, 'val'),
            transform=testTransform)
        train2 = ImageFolderSubset(
            included_classes=set2, root=os.path.join(data_root, 'train'),
            transform=trainTransform)
        test2 = ImageFolderSubset(
            included_classes=set2, root=os.path.join(data_root, 'val'),
            transform=testTransform)
    else:
        cifar_cls = dset.CIFAR10 if cifar10 else dset.CIFAR100
        train_set = cifar_cls(
            root=data_root, train=True, download=download, transform=trainTransform)
        train1 = SplitCifarDataSet(train_set, set1)
        train2 = SplitCifarDataSet(train_set, set2)

        test_set = cifar_cls(
            root=data_root, train=False,
            download=download, transform=testTransform)
        test1 = SplitCifarDataSet(test_set, set1)
        test2 = SplitCifarDataSet(test_set, set2)

    trainLoader = DataLoader(
        train1, batch_size=args.batchSz, shuffle=True, **kwargs)
    testLoader = DataLoader(
        test1, batch_size=args.batchSz, shuffle=False, **kwargs)

    bin_labels = []
    for _ in range(args.binClasses):
        vals = list(set1)
        shuffle(vals)
        bin_labels.append(set(vals[:len(vals) // 2]))

    if args.preTrainedModel:
        fname = args.preTrainedModel
    else:
        trainF = open(os.path.join(args.save, 'train1.csv'), 'w')
        testF = open(os.path.join(args.save, 'test1.csv'), 'w')

        best_error = 100
        # best_state = net.state_dict()
        ts0 = time.perf_counter()
        for epoch in range(1, 175 + 1):
            adjust_opt(args.opt, optimizer, epoch)
            train(args, epoch, net, trainLoader, optimizer, trainF, bin_labels)
            err = test(args, epoch, net, testLoader, optimizer, testF, bin_labels)

            if err < best_error:
                best_error = err
                best_state = net.state_dict()
                print('New best error {}'.format(err))
                torch.save(best_state, os.path.join(args.save, 'model_cifar{}_base.t7'.format(args.cifar)))

            # os.system('./plot.py {} &'.format(args.save))

        trainF.close()
        testF.close()
        print('Done in {:.2f}s'.format(time.perf_counter() - ts0))
        fname = os.path.join(args.save, 'model_cifar{}_base.t7'.format(args.cifar))

    return train2, test2, fname

def run_transfer_dset_b(args, ft_blocks, train2, test2, filename):
    print('Start transfer training with ft={}'.format(ft_blocks))
    cifar10 = args.cifar == 10

    if args.imagenet:
        net = DenseNetVision(
            growth_rate=32, block_config=[6, 12, 24, 16],
            num_classes=1000, num_init_features=64,
            n_binary_class=args.binClasses, binary_only=args.binWeight == 1.)
    else:
        net = densenet.DenseNet(
            growthRate=12, depth=100, reduction=0.5,
            bottleneck=True, nClasses=(10 if cifar10 else 100),
            n_binary_class=args.binClasses, binary_only=args.binWeight == 1.
        )
    if args.cuda:
        net = net.cuda()

    state = net.state_dict()
    state.update(torch.load(filename))
    net.load_state_dict(state)
    if ft_blocks == 'all':
        net.reset_final_layer()
        ft_params, reset_params = net.split_final_params()
    else:
        net.reset_layers(ft_blocks)
        ft_params, reset_params = net.split_transfer_params(ft_blocks)

    if ft_blocks:
        param_vals = [
            {'params': reset_params, 'lr': 1e-1},
            {'params': ft_params, 'lr': 1e-2}
        ]
        opt_func = adjust_opt_transfer
        epochs = 100
    else:
        param_vals = reset_params + ft_params
        opt_func = adjust_opt_transfer_baseline
        epochs = 275

    if ft_blocks != 'all':
        items = []
        for block in ft_blocks:
            if isinstance(block, tuple):
                block = '{}={}'.format(*block)
            items.append(block)
        experiment = ','.join(map(str, items))
    else:
        experiment = ft_blocks

    optimizer = optim.SGD(param_vals, lr=1e-1, momentum=0.9, weight_decay=1e-4)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    trainLoader = DataLoader(
        train2, batch_size=args.batchSz, shuffle=True, **kwargs)
    testLoader = DataLoader(
        test2, batch_size=args.batchSz, shuffle=False, **kwargs)

    trainF = open(os.path.join(args.save, 'train_ft=[{}].csv'.format(experiment)), 'w')
    testF = open(os.path.join(args.save, 'test_ft=[{}].csv'.format(experiment)), 'w')

    best_error = 100
    best_state = net.state_dict()
    ts0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        opt_func(args.opt, optimizer, epoch)
        train(args, epoch, net, trainLoader, optimizer, trainF, [])
        err = test(args, epoch, net, testLoader, optimizer, testF, [])

        if err < best_error:
            best_error = err
            best_state = net.state_dict()
            print('New best error {}'.format(err))
            torch.save(
                best_state,
                os.path.join(args.save, 'model_cifar{}_trans_ft=[{}].t7'.format(args.cifar, experiment)))

        # os.system('./plot.py {} &'.format(args.save))

    trainF.close()
    testF.close()
    print('Done in {:.2f}s'.format(time.perf_counter() - ts0))


def train(args, epoch, net, trainLoader, optimizer, trainF, bin_labels):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    ts0 = time.perf_counter()
    bin_weight = args.binWeight * 1 / len(bin_labels) if bin_labels else 0
    fc_weight = 1. - args.binWeight
    binary_only = args.binWeight == 1
    # if bin_labels:
    #     n = 2 * len(bin_labels[0])
    #     bin_weight *= 2 * (n - 1) / n

    for batch_idx, (data, target) in enumerate(trainLoader):
        ts0_batch = time.perf_counter()
        target_cls = target.__class__
        target0 = target

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        targets = [target]
        for unit_labels in bin_labels:
            labels = [1 if label in unit_labels else 0 for label in target0]
            labels = target_cls(labels)
            if args.cuda:
                labels = labels.cuda()
            targets.append(Variable(labels))

        optimizer.zero_grad()
        output = net(data)
        if args.binClasses and not bin_labels:
            output = output[0]

        if bin_labels:
            if binary_only:
                loss = F.nll_loss(output[0], targets[1]) * bin_weight
                for bin_output, bin_target in zip(output[1:], targets[2:]):
                    loss = loss + F.nll_loss(bin_output, bin_target) * bin_weight
            else:
                loss = F.nll_loss(output[0], targets[0]) * fc_weight
                for bin_output, bin_target in zip(output[1:], targets[1:]):
                    loss = loss + F.nll_loss(bin_output, bin_target) * bin_weight
        else:
            loss = F.nll_loss(output, target)

        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)

        if bin_labels:
            errors = []
            s = 1 if binary_only else 0
            fc_err = 0
            for i, (bin_output, bin_target) in enumerate(zip(output, targets[s:])):
                pred = bin_output.data.max(1)[1]  # get the index of the max log-probability
                incorrect = pred.ne(bin_target.data).cpu().sum()
                if not binary_only and not i:
                    errors.append(incorrect / len(data) * 100 * fc_weight)
                    fc_err = incorrect / len(data) * 100
                else:
                    errors.append(incorrect / len(data) * 100 * bin_weight)
            err = sum(errors)
        else:
            pred = output.data.max(1)[1] # get the index of the max log-probability
            incorrect = pred.ne(target.data).cpu().sum()
            fc_err = err = 100.*incorrect/len(data)

        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        te = time.perf_counter()
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tTime: [{:.2f}s/{:.2f}s]\tLoss: {:.6f}\tError: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            te - ts0_batch, te - ts0, loss.data[0], err))

        trainF.write('{},{},{},{}\n'.format(partialEpoch, loss.data[0], err, fc_err))
        trainF.flush()


def test(args, epoch, net, testLoader, optimizer, testF, bin_labels):
    net.eval()
    test_loss = 0
    incorrect = 0
    fc_incorrect = 0

    bin_weight = args.binWeight * 1 / len(bin_labels) if bin_labels else 0
    fc_weight = 1. - args.binWeight
    binary_only = args.binWeight == 1.

    ts0 = time.perf_counter()
    for data, target in testLoader:
        target_cls = target.__class__
        target0 = target

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        bin_targets = [target]
        for unit_labels in bin_labels:
            labels = [1 if label in unit_labels else 0 for label in target0]
            labels = target_cls(labels)
            if args.cuda:
                labels = labels.cuda()
            bin_targets.append(Variable(labels))

        output = net(data)
        if args.binClasses and not bin_labels:
            output = output[0]

        if bin_labels:
            s = 1 if binary_only else 0
            weight = fc_weight if not binary_only else bin_weight
            test_loss += F.nll_loss(output[0], bin_targets[s]).data[0] * weight
            for bin_output, bin_target in zip(output[1:], bin_targets[s + 1:]):
                test_loss += F.nll_loss(bin_output, bin_target).data[0] * bin_weight

            for i, (bin_output, bin_target) in enumerate(zip(output, bin_targets[s:])):
                pred = bin_output.data.max(1)[1]  # get the index of the max log-probability
                diff = pred.ne(bin_target.data).cpu().sum()
                if not binary_only and not i:
                    incorrect += diff * fc_weight
                    fc_incorrect += diff

                else:
                    incorrect += diff * bin_weight
        else:
            test_loss += F.nll_loss(output, target).data[0]
            pred = output.data.max(1)[1] # get the index of the max log-probability
            incorrect += pred.ne(target.data).cpu().sum()
            fc_incorrect = incorrect

    test_loss = test_loss
    test_loss /= len(testLoader) # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 100.*incorrect/nTotal
    print('\nTest set: Time: {:.2f}s, Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        time.perf_counter() - ts0, test_loss, incorrect, nTotal, err))

    testF.write('{},{},{},{}\n'.format(epoch, test_loss, err, 100 * fc_incorrect / nTotal))
    testF.flush()
    return err


def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
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


def adjust_opt_transfer(optAlg, optimizer, epoch):
    fc, base = optimizer.param_groups
    if epoch == 51:
        fc['lr'] = base['lr'] = 1e-2
    elif epoch == 76:
        fc['lr'] = base['lr'] = 1e-3


def adjust_opt_transfer_baseline(optAlg, optimizer, epoch):
    if epoch == 126:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-2
    elif epoch == 226:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-3


if __name__=='__main__':
    main()
