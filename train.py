#!/usr/bin/env python3

from functools import partial
import argparse
from PIL import Image
import torch
import copy
import numpy as np
import random
import time
from random import shuffle
import shutil
from collections import defaultdict
import os
from functools import partial

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from densenet_vision import DenseNet as DenseNetVision
from wrn import WideResNet
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset

import os
import sys
import math

import shutil

import densenet
from attic.label_cifar import unnatural_labels, natural_labels
import make_graph

iNat = {
    5, 6, 7, 15, 27, 50, 55, 73, 76, 80, 96, 104, 105, 106, 107, 130, 135, 152, 179, 185, 188, 208, 248,
    252, 264, 278, 314, 317, 330, 338, 355, 360, 371, 389, 397, 402, 404, 409, 413, 416, 433, 434, 456,
    471, 478, 481, 490, 491, 519, 530, 534, 535, 546, 555, 558, 566, 579, 581, 600, 609, 628, 634, 652,
    654, 659, 672, 681, 686, 688, 690, 693, 700, 708, 723, 746, 753, 769, 775, 802, 808, 814, 816, 837,
    858, 867, 875, 877, 881, 882, 919, 934, 938, 941, 955, 961, 971, 976, 986, 987, 996}


class ReducedDataSet(Dataset):

    dataset = None

    dataset_indices = []

    target_map = {}

    def __init__(self, dataset, cls_size=None, min_cls_size=None, min_classes=0, max_classes=0):
        self.dataset = dataset

        classes = defaultdict(list)
        for i, (img, cls) in enumerate(dataset):
            if img.size[0] > 224 and img.size[1] > 224:
                classes[cls].append(i)

        if min_cls_size:
            small_classes = {}
            for k, v in list(classes.items()):
                if len(v) < min_cls_size:
                    small_classes[k] = v
                    del classes[k]

            if min_classes and len(classes) < min_classes:
                raise Exception('Only {} classes have {} examples'.format(len(classes), min_cls_size))
                items = sorted(
                    small_classes.items(), key=lambda x: len(x[1]), reverse=True)
                for k, v in items[:min_classes - len(classes)]:
                    classes[k] = v

        dataset_indices = self.dataset_indices = []
        values = list(classes.items())
        if max_classes:
            shuffle(values)
            values = values[:max_classes]
            print('Keeping classes {}'.format([k[0] for k in values]))
        for _, indices in values:
            shuffle(indices)
            if cls_size is None:
                dataset_indices.extend(indices)
            else:
                dataset_indices.extend(indices[:cls_size])
        self.target_map = {item[0]: i for i, item in enumerate(values)}

    def __getitem__(self, index):
        if not self.dataset_indices:
            return self.dataset[index]
        img, cls = self.dataset[self.dataset_indices[index]]
        return img, self.target_map[cls]

    def __len__(self):
        return len(self.dataset_indices)


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


class PartialDataset(Dataset):

    dataset = None

    indices = []

    transform = None

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.dataset[self.indices[index]]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.indices)


def split_train_val(dataset, val_size, train_transform=None, val_transform=None):
    classes = defaultdict(list)
    for i, (_, cls) in enumerate(dataset):
        classes[cls].append(i)

    train, val = [], []
    for v in classes.values():
        shuffle(v)
        train.extend(v[:-val_size])
        val.extend(v[-val_size:])

    return PartialDataset(dataset, train,
                          transform=train_transform), PartialDataset(dataset, val, transform=val_transform)


def get_inat_dataset_stats(dataset):
    # return [0.47960037, 0.49699566, 0.41930383], [0.23362727, 0.22826615, 0.2632432]
    N = len(dataset)
    print('Computing stats for {} examples'.format(N))
    r, g, b = [], [], []
    for i, (img, _) in enumerate(dataset):
        r.append(np.asarray(img, dtype=np.uint8)[:, :, 0].ravel())
        g.append(np.asarray(img, dtype=np.uint8)[:, :, 1].ravel())
        b.append(np.asarray(img, dtype=np.uint8)[:, :, 2].ravel())

    means = []
    stdevs = []
    for c in (r, g, b):
        pixels = np.concatenate(c)
        pixels = pixels.astype(dtype=np.float32) / 255
        means.append(float(np.mean(pixels)))
        stdevs.append(float(np.std(pixels)))
    print('got {}, {}'.format(means, stdevs))
    return means, stdevs


class SplitCifarDataSet(Dataset):

    dataset = None

    data = None

    labels = None

    def __init__(self, dataset, classes, train_classes_size={}):
        self.dataset = dataset

        if dataset.train:
            orig_data = dataset.train_data
            orig_labels = dataset.train_labels
        else:
            orig_data = dataset.test_data
            orig_labels = dataset.test_labels
            train_classes_size = {}
        orig_labels = np.array(orig_labels, dtype=np.int32)

        classes = set(classes)
        select = np.zeros((len(orig_labels), ), dtype=bool)
        for i, label in enumerate(orig_labels):
            select[i] = label in classes

        for cls, size in train_classes_size.items():
            indices,  = np.nonzero(orig_labels == cls)
            np.random.shuffle(indices)
            select[indices[size:]] = False

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
    parser.add_argument('--nFTEpochs', type=int, default=100)
    parser.add_argument('--trans', action='store_true')
    parser.add_argument('--transBlocks', action='store_true')
    parser.add_argument('--transBlocksLayers', action='store_true')
    parser.add_argument('--nTransFTBlockLayersStep', type=int, default=2)
    parser.add_argument('--transFTBlock', type=int, default=0)
    parser.add_argument('--transNatSplit', action='store_true')
    parser.add_argument('--transSplit', type=int, default=50)
    parser.add_argument('--binClasses', type=int, default=0)
    parser.add_argument('--binWeight', type=float, default=.67)
    parser.add_argument('--binWeightDecay', action='store_true')
    parser.add_argument('--imagenet', action='store_true')
    parser.add_argument('--tinyImagenet', action='store_true')
    parser.add_argument('--inat', action='store_true')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--noRetrainAll', action='store_true')
    parser.add_argument('--ftSVHN', type=str, default='')
    parser.add_argument('--ftINat', type=str, default='')
    parser.add_argument('--ftCopySubset', type=str, default='')
    parser.add_argument('--ftCIFAR10', action='store_true')
    parser.add_argument('--trainAOnly', action='store_true')
    parser.add_argument('--wrn', action='store_true')
    parser.add_argument('--freezeReduce', action='store_true')
    parser.add_argument('--ftReduceEpochs', action='store_true')
    parser.add_argument('--inatNClasses', type=int, default=50)
    parser.add_argument('--imgnetNClasses', type=int, default=50)
    parser.add_argument('--dropBinaryAt', type=int, default=0)
    parser.add_argument('--limitTransClsSize', type=int, default=0)
    parser.add_argument('--dataRoot')
    parser.add_argument('--classes')
    parser.add_argument('--maml', action='store_true')
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

    if use_imagenet or args.inat:
        net = DenseNetVision(
            growth_rate=32, block_config=[6, 12, 24, 16],
            num_classes=1000, num_init_features=64,
            n_binary_class=args.binClasses, binary_only=args.binWeight == 1.)
    else:
        if args.wrn:
            net = WideResNet(depth=28, num_classes=100, widen_factor=10, dropRate=.3,
                             n_binary_class=args.binClasses)
        else:
            net = densenet.DenseNet(
                growthRate=12, depth=100, reduction=0.5,
                bottleneck=True, nClasses=(10 if cifar10 and not args.tinyImagenet else 100),
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
    elif args.inat:
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
    elif args.tinyImagenet:
        normMean =  [0.485, 0.456, 0.406]
        normStd = [0.229, 0.224, 0.225]

        normTransform = transforms.Normalize(normMean, normStd)

        trainTransform = transforms.Compose([
            transforms.RandomSizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normTransform
        ])
        testTransform = transforms.Compose([
            transforms.RandomSizedCrop(64),
            transforms.ToTensor(),
            normTransform
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
            if not args.noRetrainAll:
                run_transfer_dset_b(args, [], *res)
            run_transfer_dset_b(args, [1], *res)
            run_transfer_dset_b(args, [1, 2], *res)

        if args.transFTBlock:
            block = args.transFTBlock
            blocks = list(range(1, block + 1))
            for i in range(0, 16, args.nTransFTBlockLayersStep):
                blocks[-1] = (block, i + 1)
                run_transfer_dset_b(args, blocks, *res)
        elif args.transBlocksLayers:
            run_transfer_dset_b(args, [], *res)
            for block in range(1, 4):
                blocks = list(range(1, block + 1))
                for i in range(0, 15, args.nTransFTBlockLayersStep):
                    blocks[-1] = (block, i + 1)
                    run_transfer_dset_b(args, blocks, *res)

                blocks = list(range(1, block + 1))
                if block == 3:
                    run_transfer_dset_b(args, 'all', *res)
                else:
                    run_transfer_dset_b(args, blocks, *res)
        elif not args.trainAOnly:
            run_transfer_dset_b(args, 'all', *res)
    else:
        run(args, optimizer, net, trainTransform, testTransform)


def run(args, optimizer, net, trainTransform, testTransform):
    data_root = args.dataRoot or 'cifar'
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    if args.imagenet:
        classes = list(range(1000))
        shuffle(classes)
        train_set = ImageFolderSubset(
            included_classes=classes[:100], root=os.path.join(data_root, 'train'),
            transform=trainTransform)
        test_set = ImageFolderSubset(
            included_classes=classes[:100], root=os.path.join(data_root, 'val'),
            transform=testTransform)
    elif args.tinyImagenet:
        classes = list(range(200))
        shuffle(classes)
        train_set = ImageFolderSubset(
            included_classes=classes[:100], root=os.path.join(data_root, 'train'),
            transform=trainTransform)
        test_set = ImageFolderSubset(
            included_classes=classes[:100], root=os.path.join(data_root, 'val'),
            transform=testTransform)
    elif args.inat:
        whole_set = dset.folder.ImageFolder(root=data_root)
        whole_set = ReducedDataSet(dataset=whole_set, cls_size=650, min_cls_size=650, min_classes=100)
        train_set, test_set = split_train_val(
            whole_set, val_size=50, train_transform=trainTransform, val_transform=testTransform)
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

        torch.save(optimizer.state_dict(), os.path.join(args.save, 'optimizer_last_epoch.t7'))
        torch.save(net.state_dict(), os.path.join(args.save, 'model_last_epoch.t7'))

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
    if args.imagenet:
        n_classes = 2 * args.imgnetNClasses
        n_all = 1000
    elif args.inat:
        raise NotImplementedError
    elif args.tinyImagenet:
        n_classes = 100
        n_all = 200
    else:
        n_all = n_classes = args.cifar
    download = not args.dataRoot
    data_root = args.dataRoot or 'cifar'

    if args.transNatSplit:
        set1, set2 = natural_labels, unnatural_labels
    else:
        if args.classes:
            with open(args.classes, 'r') as fh:
                classes = list(map(int, fh.read().split(',')))
        else:
            classes = list(range(n_all))
            shuffle(classes)
            classes = classes[:n_classes]
            with open(os.path.join(args.save, 'class_shuffled'), 'w') as fh:
                fh.write(','.join(map(str, classes)))
        set1, set2 = classes[:args.transSplit], classes[args.transSplit:]

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    if args.imagenet or args.tinyImagenet:
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

        cls_limit = {}
        if args.limitTransClsSize:
            for cls in set2:
                cls_limit[cls] = args.limitTransClsSize

        train1 = SplitCifarDataSet(train_set, set1)
        train2 = SplitCifarDataSet(train_set, set2, train_classes_size=cls_limit)

        test_set = cifar_cls(
            root=data_root, train=False,
            download=download, transform=testTransform)
        test1 = SplitCifarDataSet(test_set, set1)
        test2 = SplitCifarDataSet(test_set, set2)

    if args.ftSVHN:
        train2 = dset.svhn.SVHN(
            root=args.ftSVHN, split='train', download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        )
        test2 = dset.svhn.SVHN(
            root=args.ftSVHN, split='test', download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        )
        if args.limitTransClsSize:
            train2 = ReducedDataSet(train2, cls_size=args.limitTransClsSize)
            test2 = ReducedDataSet(test2, cls_size=args.limitTransClsSize)
    elif args.ftCIFAR10:
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

        train2 = dset.CIFAR10(
            root=data_root, train=True, download=download, transform=trainTransform)
        test2 = dset.CIFAR10(
            root=data_root, train=False,
            download=download, transform=testTransform)
        if args.limitTransClsSize:
            train2 = ReducedDataSet(train2, cls_size=args.limitTransClsSize)
            test2 = ReducedDataSet(test2, cls_size=args.limitTransClsSize)
    elif args.ftINat:
        if os.path.isdir(os.path.join(args.ftINat, 'train')):
            normTransform = transforms.Normalize([0.481523334980011, 0.5006749033927917, 0.4647780954837799],
                                                 [0.2249356061220169, 0.22407741844654083, 0.2626110017299652])
            trainTransform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normTransform
            ])
            testTransform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                normTransform
            ])
            train2 = dset.folder.ImageFolder(root=os.path.join(args.ftINat, 'train'), transform=trainTransform)
            test2 = dset.folder.ImageFolder(root=os.path.join(args.ftINat, 'val'), transform=testTransform)
        else:
            orig = whole_set = dset.folder.ImageFolder(root=args.ftINat)
            whole_set = ReducedDataSet(dataset=whole_set, cls_size=(args.limitTransClsSize or 600) + 50,
                                       min_cls_size=(args.limitTransClsSize or 600) + 50, min_classes=args.inatNClasses,
                                       max_classes=args.inatNClasses)
            normTransform = transforms.Normalize(*get_inat_dataset_stats(whole_set))
            trainTransform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normTransform
            ])
            testTransform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                normTransform
            ])
            train2, test2 = split_train_val(
                whole_set, val_size=50, train_transform=trainTransform, val_transform=testTransform)
            if args.ftCopySubset:
                for name, dataset in [('train', train2), ('val', test2)]:
                    for idx in dataset.indices:
                        path = orig.imgs[whole_set.dataset_indices[idx]][0]
                        new_path = path.replace(args.ftINat, os.path.join(args.ftCopySubset, name))
                        if not os.path.exists(os.path.dirname(new_path)):
                            os.makedirs(os.path.dirname(new_path))
                        shutil.copy2(path, new_path)

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
        train_fn = train_maml if args.maml else train
        opt = adjust_opt_wrn if args.wrn else adjust_opt
        for epoch in range(1, args.nEpochs + 1):
            opt(args.opt, optimizer, epoch)
            train_fn(args, epoch, net, trainLoader, optimizer, trainF, bin_labels)
            err = test(args, epoch, net, testLoader, optimizer, testF, bin_labels)

            torch.save(optimizer.state_dict(), os.path.join(args.save, 'optimizer_last_epoch.t7'))
            torch.save(net.state_dict(), os.path.join(args.save, 'model_last_epoch.t7'))

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

    if args.imagenet or args.inat:
        net = DenseNetVision(
            growth_rate=32, block_config=[6, 12, 24, 16],
            num_classes=1000, num_init_features=64,
            n_binary_class=args.binClasses, binary_only=args.binWeight == 1.)
    else:
        if args.wrn:
            net = WideResNet(depth=28, num_classes=100, widen_factor=10, dropRate=.3,
                             n_binary_class=args.binClasses)
        else:
            net = densenet.DenseNet(
                growthRate=12, depth=100, reduction=0.5,
                bottleneck=True, nClasses=(10 if cifar10 and not args.tinyImagenet else 100),
                n_binary_class=args.binClasses, binary_only=args.binWeight == 1.
            )
    if args.cuda:
        net = net.cuda()

    state = net.state_dict()
    state.update(torch.load(filename))
    net.load_state_dict(state)

    if args.freezeReduce:
        if ft_blocks == 'all':
            net.freeze_layers([1, 2, 3])
        else:
            net.freeze_layers(ft_blocks)

        net.reset_final_layer()
        ft_params, reset_params = net.split_final_params()
        ft_params = list(filter(lambda p: p.requires_grad, ft_params))
    elif ft_blocks == 'all':
        net.reset_final_layer()
        ft_params, reset_params = net.split_final_params()
    else:
        net.reset_layers(ft_blocks)
        ft_params, reset_params = net.split_transfer_params(ft_blocks)

    if ft_blocks or args.freezeReduce or args.ftReduceEpochs:
        if ft_params:
            param_vals = [
                {'params': reset_params, 'lr': 1e-1},
                {'params': ft_params, 'lr': 1e-2}
            ]
        else:
            param_vals = [{'params': reset_params, 'lr': 1e-1}]
        if args.freezeReduce or args.ftReduceEpochs:
            epochs = 40
            opt_func = partial(adjust_opt_transfer, epoch1=26, epoch2=36)
        else:
            opt_func = adjust_opt_transfer
            epochs = args.nFTEpochs
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
    if args.dropBinaryAt and args.dropBinaryAt <= epoch:
        bin_labels = []

    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    ts0 = time.perf_counter()
    bin_decay = args.binWeightDecay
    bin_weight = args.binWeight * (1 if bin_decay else 1 / len(bin_labels)) if bin_labels else 0
    fc_weight = 1. - args.binWeight
    binary_only = args.binWeight == 1
    if bin_decay and len(bin_labels) > 1:
        bin_weights = list(reversed(list(range(1, len(bin_labels) + 1))))
        weight_sum = sum(bin_weights)
        bin_weights = [w / weight_sum for w in bin_weights]
    else:
        bin_weights = [1 for _ in bin_labels]
    if binary_only and args.dropBinaryAt:
        raise Exception('Cannot have binary only and early binary dropping')

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
        if args.binClasses and not bin_labels:  # fine tuning model with bin
            output = output[0]

        if bin_labels:
            if binary_only:
                loss = F.nll_loss(output[0], targets[1]) * bin_weight * bin_weights[0]
                for bin_output, bin_target, w in zip(output[1:], targets[2:], bin_weights[1:]):
                    loss = loss + F.nll_loss(bin_output, bin_target) * bin_weight * w
            else:
                loss = F.nll_loss(output[0], targets[0]) * fc_weight
                for bin_output, bin_target, w in zip(output[1:], targets[1:], bin_weights):
                    loss = loss + F.nll_loss(bin_output, bin_target) * bin_weight * w

            errors = []
            s = 1 if binary_only else 0
            fc_err = 0
            w_iter = iter(bin_weights)
            for i, (bin_output, bin_target) in enumerate(zip(output, targets[s:])):
                pred = bin_output.data.max(1)[1]  # get the index of the max log-probability
                incorrect = pred.ne(bin_target.data).cpu().sum()
                if not binary_only and not i:
                    errors.append(incorrect / len(data) * 100 * fc_weight)
                    fc_err = incorrect / len(data) * 100
                else:
                    errors.append(incorrect / len(data) * 100 * bin_weight * next(w_iter))
            err = sum(errors)
        else:
            loss = F.nll_loss(output, target)

            pred = output.data.max(1)[1] # get the index of the max log-probability
            incorrect = pred.ne(target.data).cpu().sum()
            fc_err = err = 100.*incorrect/len(data)

        del output
        loss.backward()
        optimizer.step()
        nProcessed += len(data)

        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        te = time.perf_counter()
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tTime: [{:.2f}s/{:.2f}s]\tLoss: {:.6f}\tError: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            te - ts0_batch, te - ts0, loss.data[0], err))

        trainF.write('{},{},{},{}\n'.format(partialEpoch, loss.data[0], err, fc_err))
        trainF.flush()


def _replace_val(newval, grad):
    return newval


def train_maml(args, epoch, net, trainLoader, optimizer, trainF, bin_labels):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    ts0 = time.perf_counter()

    bin_weight = args.binWeight * 1 / len(bin_labels) if bin_labels else 0
    fc_weight = 1. - args.binWeight
    binary_only = args.binWeight == 1

    lrs = [param_group['lr'] for param_group in optimizer.param_groups]
    bin_data_iters = [(iter(trainLoader), iter(trainLoader)) for _ in bin_labels]
    fc_iter1, fc_iter2 = iter(trainLoader), iter(trainLoader)

    dummy_data, dummy_target = next(iter(trainLoader))
    target_cls = dummy_target.__class__
    # dummy_labels = target_cls([0, ] * len(dummy_target))
    # if args.cuda:
    #     dummy_target_var = Variable(dummy_labels.cuda())
    #     dummy_data_var = Variable(dummy_data.cuda())
    # else:
    #     dummy_target_var = Variable(dummy_labels)
    #     dummy_data_var = Variable(dummy_data)

    batch_idx = 0
    done = False
    while not done:
        ts0_batch = time.perf_counter()
        task_grads = []
        optimizer.zero_grad()
        original_state = copy.deepcopy(net.state_dict())
        errors = []

        for classifier, unit_labels, (iter1, iter2) in zip(net.binary_layers, bin_labels, bin_data_iters):
            try:
                data, target = next(iter1)
            except StopIteration:
                done = True
                break

            labels = [1 if label in unit_labels else 0 for label in target]
            labels = target_cls(labels)
            if args.cuda:
                data, labels = data.cuda(), labels.cuda()
            data, labels = Variable(data), Variable(labels)

            optim_state = copy.deepcopy(optimizer.state_dict())

            optimizer.zero_grad()
            output = F.log_softmax(classifier(net(data, skip_classifier=True)))
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()

            data, target = next(iter2)
            labels = [1 if label in unit_labels else 0 for label in target]
            labels = target_cls(labels)
            if args.cuda:
                data, labels = data.cuda(), labels.cuda()
            data, labels = Variable(data), Variable(labels)

            optimizer.zero_grad()
            output = F.log_softmax(classifier(net(data, skip_classifier=True)))
            loss = F.nll_loss(output, labels) * bin_weight

            pred = output.data.max(1)[1]  # get the index of the max log-probability
            incorrect = pred.ne(labels.data).cpu().sum()
            errors.append(incorrect / len(data) * 100 * bin_weight)

            loss.backward()
            task_grads.append([copy.deepcopy(param.grad) for param in net.parameters()])

            net.load_state_dict(original_state)
            optimizer.load_state_dict(optim_state)
            # XXX: https://discuss.pytorch.org/t/saving-and-loading-sgd-optimizer/2536
            optimizer.state = defaultdict(dict, optimizer.state)

        if done:
            break

        fc_err = 0
        if not binary_only:
            optim_state = copy.deepcopy(optimizer.state_dict())

            try:
                data, target = next(fc_iter1)
            except StopIteration:
                break

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            optimizer.zero_grad()
            output = F.log_softmax(net.fc(net(data, skip_classifier=True)))
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            data, target = next(fc_iter2)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            optimizer.zero_grad()
            output = F.log_softmax(net.fc(net(data, skip_classifier=True)))
            loss = F.nll_loss(output, target) * fc_weight

            pred = output.data.max(1)[1]  # get the index of the max log-probability
            incorrect = pred.ne(target.data).cpu().sum()
            errors.append(incorrect / len(data) * 100 * fc_weight)
            fc_err = incorrect / len(data) * 100

            loss.backward()
            task_grads.append([copy.deepcopy(param.grad) for param in net.parameters()])

            net.load_state_dict(original_state)
            optimizer.load_state_dict(optim_state)
            # XXX: https://discuss.pytorch.org/t/saving-and-loading-sgd-optimizer/2536
            optimizer.state = defaultdict(dict, optimizer.state)

        # dummy step to fill in buffers so grads will be replaced
        dummy_labels = target_cls([0, ] * len(dummy_target))
        if args.cuda:
            dummy_target_var = Variable(dummy_labels.cuda())
            dummy_data_var = Variable(dummy_data.cuda())
        else:
            dummy_target_var = Variable(dummy_labels)
            dummy_data_var = Variable(dummy_data)

        hooks = []
        optimizer.zero_grad()
        for param, values in zip(net.parameters(), zip(*task_grads)):
            hooks.append(param.register_hook(partial(_replace_val, sum((v for v in values if v is not None)))))

        if bin_labels:
            base_out = net(dummy_data_var, skip_classifier=True)
            output = F.log_softmax(net.binary_layers[0](base_out))
            loss = F.nll_loss(output, dummy_target_var)
            for layer, unit_labels in zip(net.binary_layers[1:], bin_labels[1:]):
                output = F.log_softmax(layer(base_out))
                loss += F.nll_loss(output, dummy_target_var)

            if not binary_only:
                output = F.log_softmax(net.fc(base_out))
                loss += F.nll_loss(output, dummy_target_var)
            del base_out
        else:
            output = F.log_softmax(net.fc(net(dummy_data_var, skip_classifier=True)))
            loss = F.nll_loss(output, dummy_target_var)
        del output
        loss.backward()

        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = .001
        optimizer.step()
        for h in hooks:
            h.remove()
        # for param_group, lr in zip(optimizer.param_groups, lrs):
        #     param_group['lr'] = lr

        err = sum(errors)
        nProcessed += len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        te = time.perf_counter()
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tTime: [{:.2f}s/{:.2f}s]\tLoss: {:.6f}\tError: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            te - ts0_batch, te - ts0, loss.data[0], err))

        trainF.write('{},{},{},{}\n'.format(partialEpoch, loss.data[0], err, fc_err))
        trainF.flush()

        batch_idx += 1


def test(args, epoch, net, testLoader, optimizer, testF, bin_labels):
    net.eval()
    if args.dropBinaryAt and args.dropBinaryAt <= epoch:
        bin_labels = []
    test_loss = 0
    incorrect = 0
    fc_incorrect = 0

    bin_decay = args.binWeightDecay
    bin_weight = args.binWeight * (1 if bin_decay else 1 / len(bin_labels)) if bin_labels else 0
    fc_weight = 1. - args.binWeight
    binary_only = args.binWeight == 1.
    if bin_decay and len(bin_labels) > 1:
        bin_weights = list(reversed(list(range(1, len(bin_labels) + 1))))
        weight_sum = sum(bin_weights)
        bin_weights = [w / weight_sum for w in bin_weights]
    else:
        bin_weights = [1 for _ in bin_labels]

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
        if args.binClasses and not bin_labels:  # fine tuning
            output = output[0]

        if bin_labels:
            w_iter = iter(bin_weights)
            s = 1 if binary_only else 0
            weight = fc_weight if not binary_only else (bin_weight * next(w_iter))
            test_loss += F.nll_loss(output[0], bin_targets[s]).data[0] * weight
            for bin_output, bin_target in zip(output[1:], bin_targets[s + 1:]):
                test_loss += F.nll_loss(bin_output, bin_target).data[0] * bin_weight * next(w_iter)

            w_iter = iter(bin_weights)
            for i, (bin_output, bin_target) in enumerate(zip(output, bin_targets[s:])):
                pred = bin_output.data.max(1)[1]  # get the index of the max log-probability
                diff = pred.ne(bin_target.data).cpu().sum()
                if not binary_only and not i:
                    incorrect += diff * fc_weight
                    fc_incorrect += diff
                else:
                    incorrect += diff * bin_weight * next(w_iter)
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


def adjust_opt_wrn(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch == 60:
            lr = 1e-1 * .2
        elif epoch == 120:
            lr = 1e-1 * (.2 ** 2)
        elif epoch == 160:
            lr = 1e-1 * (.2 ** 3)
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def adjust_opt_transfer(optAlg, optimizer, epoch, epoch1=51, epoch2=76):
    if len(optimizer.param_groups) == 2:
        fc, base = optimizer.param_groups
    else:
        fc, = optimizer.param_groups
        base = {'lr': 3}
    if epoch == epoch1:
        fc['lr'] = base['lr'] = 1e-2
    elif epoch == epoch2:
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
