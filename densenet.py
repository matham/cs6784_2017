import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    layer_funcs = {}

    binary_layers = []

    binary_only = False

    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck, n_binary_class=0,
                 binary_only=False):
        super(DenseNet, self).__init__()

        self.layer_funcs = {1: self.get_first_dense_block_layers,
                            2: self.get_second_dense_block_layers,
                            3: self.get_third_dense_block_layers}
        self.binary_only = binary_only

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        self.final_chans = nChannels
        self.n_classes = nClasses

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)
        self.binary_layers = bins = []
        for i in range(n_binary_class):
            bins.append(nn.Linear(nChannels, 2))
            setattr(self, 'bin_fc{}'.format(i), bins[-1])

        for m in self.modules():
            self.set_layer_params(m)

    def set_layer_params(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def reset_final_layer(self):
        reset_layers = [self.fc] if not self.binary_only else []
        reset_layers.extend(self.binary_layers)

        for layer in reset_layers:
            layer.reset_parameters()
            self.set_layer_params(layer)

    def reset_layers(self, ft_blocks):
        reset_layers = [self.fc] if not self.binary_only else []
        reset_layers.extend(self.binary_layers)
        blocks_skipped = {b[0] if isinstance(b, tuple) else b for b in ft_blocks}
        for i in range(1, 4):
            if i not in blocks_skipped:
                reset_layers.extend(self.layer_funcs[i]())

        for block in ft_blocks:
            if isinstance(block, tuple):
                block, layer = block
                reset_layers.extend(self.layer_funcs[block](layer))

        for layer in reset_layers:
            layer.reset_parameters()
            self.set_layer_params(layer)

    def split_transfer_params(self, ft_blocks):
        reset_layers = [self.fc] if not self.binary_only else []
        reset_layers.extend(self.binary_layers)
        blocks_skipped = {b[0] if isinstance(b, tuple) else b for b in ft_blocks}
        for i in range(1, 4):
            if i not in blocks_skipped:
                reset_layers.extend(self.layer_funcs[i]())

        for block in ft_blocks:
            if isinstance(block, tuple):
                block, layer = block
                reset_layers.extend(self.layer_funcs[block](layer))

        params = list(self.parameters())
        reset_params = []
        for layer in reset_layers:
            reset_params.extend(layer.parameters())

        ft_params = [p for p in params if not [r_p for r_p in reset_params if r_p is p]]
        return ft_params, reset_params

    def split_final_params(self):
        reset_layers = [self.fc] if not self.binary_only else []
        reset_layers.extend(self.binary_layers)

        params = list(self.parameters())
        reset_params = []
        for layer in reset_layers:
            reset_params.extend(layer.parameters())

        ft_params = [p for p in params if not [r_p for r_p in reset_params if r_p is p]]
        return ft_params, reset_params

    def get_first_dense_block_layers(self, layer=0):
        layers = [
            layer for group in list(self.dense1.children())[layer:]
            for layer in group.children()]
        if not layer:
            layers.append(self.conv1)
        for layer in self.trans1.children():
            layers.append(layer)
        return layers

    def get_second_dense_block_layers(self, layer=0):
        layers = [
            layer for group in list(self.dense2.children())[layer:]
            for layer in group.children()]
        for layer in self.trans2.children():
            layers.append(layer)
        return layers

    def get_third_dense_block_layers(self, layer=0):
        layers = [
            layer for group in list(self.dense3.children())[layer:]
            for layer in group.children()]
        layers.append(self.bn1)
        return layers

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))

        binary_out = []
        for layer in self.binary_layers:
            binary_out.append(F.log_softmax(layer(out)))

        if self.binary_only:
            return binary_out

        binary_out.insert(0, F.log_softmax(self.fc(out)))

        if len(binary_out) == 1:
            return binary_out[0]
        return binary_out
