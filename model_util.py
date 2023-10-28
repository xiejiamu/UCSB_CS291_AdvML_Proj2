# This file implements ResNet, you don't need to modify anything in this file

import time
import torch
import torch.nn as nn


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location))

    def save(self, name=None):
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def no_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def with_grad(self):
        for param in self.parameters():
            param.requires_grad = True

    def clear_grad(self):
        for param in self.parameters():
            param.grad = None


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation_fn=nn.ReLU()):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.activation_fn = activation_fn
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.activation_fn(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation_fn(out)
        return out


class ResNet(BasicModule):
    def __init__(self, block, num_blocks, num_classes=10, activation_fn=nn.ReLU, conv1_size=3):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.activation_fn = activation_fn(beta=10) if activation_fn == nn.Softplus else activation_fn()

        kernel_size, stride, padding = {3: [3, 1, 1], 7: [7, 2, 3], 15: [15, 3, 7]}[conv1_size]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, activation_fn=self.activation_fn)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, activation_fn=self.activation_fn)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, activation_fn=self.activation_fn)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, activation_fn=self.activation_fn)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.normalize = None

    def _make_layer(self, block, planes, num_blocks, stride, activation_fn=nn.ReLU()):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, activation_fn))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, penu=False):
        if self.normalize:
            x = self.normalize(x)
        out = self.activation_fn(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        if penu:
            return out
        out = self.linear(out)
        return out


def ResNet18(num_classes=10, conv1_size=3, activation_fn=nn.ReLU):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, conv1_size=conv1_size, activation_fn=activation_fn)
