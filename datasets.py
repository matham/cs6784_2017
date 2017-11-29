from __future__ import print_function
from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity


class CIFAR100(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = 'cifar-100-python'

    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

    filename = "cifar-100-python.tar.gz"

    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]

    def __init__(self, root, group, train=True,
                 transform=None, target_transform=None,
                 download=False, classes_a=[], classes_a_bin=[]):
        self.root = os.path.expanduser(root)
        self.group = group
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.classes_a_bin = classes_a_bin

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []

            for entry in self.train_list:
                f = entry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                    # print(entry['fine_labels'])
                    # raise Exception
                fo.close()

            self.train_labels = np.array(self.train_labels, dtype=np.uint8)
            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

            indices_a = []
            indices_b = []
            for i, label in enumerate(self.train_labels):
                if label in classes_a:
                    indices_a.append(i)
                else:
                    indices_b.append(i)
            indices_a = np.array(indices_a)
            indices_b = np.array(indices_b)

            self.train_dataA = self.train_data[indices_a, :, :, :]
            self.train_labelsA = self.train_labels[indices_a]

            self.train_dataB = self.train_data[indices_b, :, :, :]
            self.train_labelsB = self.train_labels[indices_b]
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']

            fo.close()

            self.test_labels = np.array(self.test_labels, dtype=np.uint8)
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

            indices_a = []
            indices_b = []
            for i, label in enumerate(self.test_labels):
                if label in classes_a:
                    indices_a.append(i)
                else:
                    indices_b.append(i)
            indices_a = np.array(indices_a)
            indices_b = np.array(indices_b)

            self.test_dataA = self.test_data[indices_a, :, :, :]
            self.test_labelsA = self.test_labels[indices_a]

            self.test_dataB = self.test_data[indices_b, :, :, :]
            self.test_labelsB = self.test_labels[indices_b]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            if self.group == 'A':
                img, target = self.train_dataA[index], int(self.train_labelsA[index])
                bin_target = 0 if target in self.classes_a_bin else 1
            elif self.group == 'B':
                img, target = self.train_dataB[index], int(self.train_labelsB[index])
        else:
            if self.group == 'A':
                img, target = self.test_dataA[index], int(self.test_labelsA[index])
                bin_target = 0 if target in self.classes_a_bin else 1
            elif self.group == 'B':
                img, target = self.test_dataB[index], int(self.test_labelsB[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            if self.group == 'A':
                bin_target = self.target_transform(bin_target)

        if self.group == 'A':
            return img, (target, bin_target)
        else:
            return img, target

    def __len__(self):
        if self.train:
            if self.group == 'A':
                return len(self.train_dataA)
            elif self.group == 'B':
                return len(self.train_dataB)
        else:
            if self.group == 'A':
                return len(self.test_dataA)
            elif self.group == 'B':
                return len(self.test_dataB)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True
