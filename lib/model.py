"""
author: Viktor Cerny
created: 03-05-19
last edit: 29-05-19
desc: all neural-network-models (classes) are found in this file
"""

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

#v 0.5.0
#desc: the analizer used for evaluating
#for the first time using ResNeXt Architecture
#code taken from: https://github.com/prlz77/ResNeXt.pytorch/blob/master/models/model.py
class ResNeXtBottleneck(nn.Module):
    def __init__(self,in_channels, out_channels, stride, cardinality, base_width, widen_factor):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck,self).__init__()
        width_ratio = out_channels / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1,stride=1,padding=0,bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D,D, kernel_size=3,stride=stride,padding=1,groups=cardinality,bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1,stride=1,padding=0,bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels,kernel_size=1,stride=stride,padding=0,
                                               bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self,x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck),inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return F.relu(residual + bottleneck, inplace=True)


class AnalizerCNN(nn.Module):
    def __init__(self, cardinality, depth, nlabels, base_width, widen_factor=4):
        """ Constructor
        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(AnalizerCNN, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth -2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]

        self.conv_1_3x3 = nn.Conv2d(3,64,3,1,1,bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1],1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2],2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3],2)
        self.classifier = nn.Linear(self.stages[3], nlabels)
        init.kaiming_normal(self.classifier.weight)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key],mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def block(self, name, in_channels, out_channels, pool_stride=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
           name: string name of the current block.
           in_channels: number of input channels
           out_channels: number of output channels
           pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, ResNeXtBottleneck(in_channels, out_channels, pool_stride,
                                                          self.cardinality, self.base_width,
                                                          self.widen_factor))
            else:
                block.add_module(name_, ResNeXtBottleneck(out_channels, out_channels, 1,
                                                          self.cardinality,self.base_width,
                                                          self.widen_factor))

        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, self.stages[3])
        return self.classifier(x)



class AnalizerCNNEval(nn.Module):
    def __init__(self, cardinality, depth, nlabels, base_width, widen_factor=4):
        """ Constructor
        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(AnalizerCNN, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth -2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]

        self.conv_1_3x3 = nn.Conv2d(3,64,3,1,1,bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1],1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2],2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3],2)
        self.classifier = nn.Linear(self.stages[3], nlabels)
        init.kaiming_normal(self.classifier.weight)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key],mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def block(self, name, in_channels, out_channels, pool_stride=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
           name: string name of the current block.
           in_channels: number of input channels
           out_channels: number of output channels
           pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, ResNeXtBottleneck(in_channels, out_channels, pool_stride,
                                                          self.cardinality, self.base_width,
                                                          self.widen_factor))
            else:
                block.add_module(name_, ResNeXtBottleneck(out_channels, out_channels, 1,
                                                          self.cardinality,self.base_width,
                                                          self.widen_factor))

        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, self.stages[3])
        return self.classifier(x)

























# #v 0.4.5
# #desc: the analizer used for evaluating
# class AnalizerCNNEval(nn.Module):
#     def __init__(self,inputSize):
#         super(AnalizerCNNEval, self).__init__()
#         self.inputSize = inputSize
#         self.endLayerSize = 256
#
#         #TODO:
#         #make Kernelsize variable and dependend on inputSize
#         #also add maxPools when variable is implemented
#         self.conv = nn.Sequential(
#                     nn.Conv2d(3, 256, 5,padding=2),         #(25x25)
#                     nn.MaxPool2d(2),                        #(12x12)
#                     nn.ReLU(),
#                     nn.Conv2d(256, 512, 3,padding=1),       #(12x12)
#                     nn.MaxPool2d(2),                        #(6x6)
#                     nn.ReLU(),
#                     nn.Conv2d(512, 512, 3,padding=1),       # (6x6)
#                     nn.MaxPool2d(2),                        # (3x3)
#                     nn.ReLU(),
#                     nn.Conv2d(512, 1024, 3),                # (1x1)
#                     nn.ReLU(),
#                     nn.Conv2d(1024, self.endLayerSize, 1),  # (1x1)
#                     nn.ReLU(),
#         )
#
#         self.linear = nn.Sequential(
#                         nn.Linear(self.endLayerSize,2000),
#                         nn.Linear(2000, 1000),
#                         nn.Linear(1000, 2121),
#         )
#
#     def forward(self, img):
#         x = self.conv(img)
#         x = x.view(self.inputSize,self.endLayerSize)
#         x = self.linear(x)
#         return x
#
# #v 0.4.5
# #desc: the analizer used for training
# class AnalizerCNN(nn.Module):
#     def __init__(self,inputSize):
#         super(AnalizerCNN, self).__init__()
#         self.inputSize = inputSize
#         self.endLayerSize = 256
#
#         #TODO:
#         #make Kernelsize variable and dependend on inputSize
#         #also add maxPools when variable is implemented
#         self.conv = nn.Sequential(
#                     nn.Conv2d(3, 256, 5,padding=2),         #(25x25)
#                     nn.MaxPool2d(2),                        #(12x12)
#                     nn.ReLU(),
#                     nn.Conv2d(256, 512, 3,padding=1),       #(12x12)
#                     nn.MaxPool2d(2),                        #(6x6)
#                     nn.ReLU(),
#                     nn.Conv2d(512, 512, 3,padding=1),       # (6x6)
#                     nn.MaxPool2d(2),                        # (3x3)
#                     nn.ReLU(),
#                     nn.Conv2d(512, 1024, 3),                # (1x1)
#                     nn.ReLU(),
#                     nn.Conv2d(1024, self.endLayerSize, 1),  # (1x1)
#                     nn.ReLU(),
#         )
#
#         self.linear = nn.Sequential(
#                         nn.Linear(self.endLayerSize,2000),
#                         nn.Linear(2000, 1000),
#                         nn.Linear(1000, 2121),
#         )
#
#     def forward(self, img):
#         x = self.conv(img)
#         x = x.view(self.inputSize,self.endLayerSize)
#         x = self.linear(x)
#         return x













# #v 0.4.4
# #desc: the analizer used for evaluating
# class AnalizerCNNEval(nn.Module):
#     def __init__(self,inputSize):
#         super(AnalizerCNNEval, self).__init__()
#         self.inputSize = inputSize
#         self.endLayerSize = 1024
#
#         #TODO:
#         #make Kernelsize variable and dependend on inputSize
#         #also add maxPools when variable is implemented
#         self.conv = nn.Sequential(
#                     nn.Conv2d(3, 256, 5),       #(21x21)
#                     nn.MaxPool2d(2),            #(10x10)
#                     nn.ReLU(),
#                     nn.Conv2d(256, 512, 3),     #(8x8)
#                     nn.MaxPool2d(2),            #(4x4)
#                     nn.ReLU(),
#                     nn.Conv2d(512, 1024, 3),     # (2x2)
#                     nn.MaxPool2d(2),            # (1x1)
#                     nn.ReLU(),
#                     nn.Conv2d(1024, self.endLayerSize, 1),  # (1x1)
#                     nn.ReLU(),
#         )
#
#         self.linear = nn.Sequential(
#                         nn.Linear(self.endLayerSize,1000),
#                         nn.Linear(1000, 2121),
#         )
#
#     def forward(self, img):
#         x = self.conv(img)
#         x = x.view(self.inputSize,self.endLayerSize)
#         x = self.linear(x)
#         return x
#
# #v 0.4.4
# #desc: the analizer used for training
# class AnalizerCNN(nn.Module):
#     def __init__(self,inputSize):
#         super(AnalizerCNN, self).__init__()
#         self.inputSize = inputSize
#         self.endLayerSize = 1024
#
#         #TODO:
#         #make Kernelsize variable and dependend on inputSize
#         #also add maxPools when variable is implemented
#         self.conv = nn.Sequential(
#                     nn.Conv2d(3, 256, 5),       #(21x21)
#                     nn.MaxPool2d(2),            #(10x10)
#                     nn.ReLU(),
#                     nn.Conv2d(256, 512, 3),     #(8x8)
#                     nn.MaxPool2d(2),            #(4x4)
#                     nn.ReLU(),
#                     nn.Conv2d(512, 1024, 3),     # (2x2)
#                     nn.MaxPool2d(2),            # (1x1)
#                     nn.ReLU(),
#                     nn.Conv2d(1024, self.endLayerSize, 1),  # (1x1)
#                     nn.ReLU(),
#         )
#
#         self.linear = nn.Sequential(
#                         nn.Linear(self.endLayerSize,1000),
#                         nn.Linear(1000, 2121),
#         )
#
#     def forward(self, img):
#         x = self.conv(img)
#         x = x.view(self.inputSize,self.endLayerSize)
#         x = self.linear(x)
#         return x








# #v 0.4.3
# #desc: the analizer used for evaluating
# class AnalizerCNNEval(nn.Module):
#     def __init__(self,inputSize):
#         super(AnalizerCNNEval, self).__init__()
#         self.inputSize = inputSize
#         self.endLayerSize = 512
#
#         #TODO:
#         #make Kernelsize variable and dependend on inputSize
#         #also add maxPools when variable is implemented
#         self.conv = nn.Sequential(
#                         nn.Conv2d(3, 128, 5),       #(21x21)
#                         nn.MaxPool2d(2),            #(10x10)?
#                         nn.ReLU(),
#                         nn.Conv2d(128, 256, 3),     #(8x8)
#                         nn.MaxPool2d(2),            #(4x4)
#                         nn.ReLU(),
#                         nn.Conv2d(256, self.endLayerSize, 4),  # (1x1)
#                         nn.ReLU(),
#         )
#
#         self.linear = nn.Sequential(
#                         nn.Linear(self.endLayerSize,1000),
#                         nn.Linear(1000, 2121),
#         )
#
#     def forward(self, img):
#         x = self.conv(img)
#         x = x.view(self.inputSize,self.endLayerSize)
#         x = self.linear(x)
#         return x
#
# #v 0.4.3
# #desc: the analizer used for training
# class AnalizerCNN(nn.Module):
#     def __init__(self,inputSize):
#         super(AnalizerCNN, self).__init__()
#         self.inputSize = inputSize
#         self.endLayerSize = 512
#
#         #TODO:
#         #make Kernelsize variable and dependend on inputSize
#         #also add maxPools when variable is implemented
#         self.conv = nn.Sequential(
#                     nn.Conv2d(3, 128, 5),       #(21x21)
#                     nn.MaxPool2d(2),            #(10x10)?
#                     nn.ReLU(),
#                     nn.Conv2d(128, 256, 3),     #(8x8)
#                     nn.MaxPool2d(2),            #(4x4)
#                     nn.ReLU(),
#                     nn.Conv2d(256, self.endLayerSize, 4),  # (1x1)
#                     nn.ReLU(),
#         )
#
#         self.linear = nn.Sequential(
#                         nn.Linear(self.endLayerSize,1000),
#                         nn.Linear(1000, 2121),
#         )
#
#     def forward(self, img):
#         x = self.conv(img)
#         x = x.view(self.inputSize,self.endLayerSize)
#         x = self.linear(x)
#         return x





# #v 0.4.2
# #desc: the analizer used for evaluating
# class AnalizerCNNEval(nn.Module):
#     def __init__(self,inputSize):
#         super(AnalizerCNNEval, self).__init__()
#         self.inputSize = inputSize
#         self.endLayerSize = 256
#
#         #TODO:
#         #make Kernelsize variable and dependend on inputSize
#         #also add maxPools when variable is implemented
#         self.conv = nn.Sequential(
#                     nn.Conv2d(3, 64, 5),       #(21x21)
#                     nn.MaxPool2d(2),            #(10x10)?
#                     nn.ReLU(),
#                     nn.Conv2d(64, 128, 3),     #(8x8)
#                     nn.MaxPool2d(2),            #(4x4)
#                     nn.ReLU(),
#                     nn.Conv2d(128, 256, 4),  # (1x1)
#                     nn.ReLU(),
#         )
#
#         self.linear = nn.Sequential(
#                         nn.Linear(self.endLayerSize,1000),
#                         nn.Linear(1000, 2121),
#         )
#
#     def forward(self, img):
#         x = self.conv(img)
#         x = x.view(self.inputSize,self.endLayerSize)
#         x = self.linear(x)
#         return x
#
# #v 0.4.2
# #desc: the analizer used for training
# class AnalizerCNN(nn.Module):
#     def __init__(self,inputSize):
#         super(AnalizerCNN, self).__init__()
#         self.inputSize = inputSize
#         self.endLayerSize = 256
#
#         #TODO:
#         #make Kernelsize variable and dependend on inputSize
#         #also add maxPools when variable is implemented
#         self.conv = nn.Sequential(
#                     nn.Conv2d(3, 64, 5),       #(21x21)
#                     nn.MaxPool2d(2),            #(10x10)?
#                     nn.ReLU(),
#                     nn.Conv2d(64, 128, 3),     #(8x8)
#                     nn.MaxPool2d(2),            #(4x4)
#                     nn.ReLU(),
#                     nn.Conv2d(128, 256, 4),  # (1x1)
#                     nn.ReLU(),
#         )
#
#         self.linear = nn.Sequential(
#                         nn.Linear(self.endLayerSize,1000),
#                         nn.Linear(1000, 2121),
#         )
#
#     def forward(self, img):
#         x = self.conv(img)
#         x = x.view(self.inputSize,self.endLayerSize)
#         x = self.linear(x)
#         return x





# #v 0.4.1
# #desc: the analizer used for evaluating
# class AnalizerCNNEval(nn.Module):
#     def __init__(self,inputSize):
#         super(AnalizerCNNEval, self).__init__()
#         self.inputSize = inputSize
#         self.endLayerSize = 256
#
#         #TODO:
#         #make Kernelsize variable and dependend on inputSize
#         #also add maxPools when variable is implemented
#         self.conv = nn.Sequential(
#                     nn.Conv2d(3, 64, 5),       #(21x21)
#                     nn.MaxPool2d(2),            #(10x10)?
#                     nn.ReLU(),
#                     nn.Conv2d(64, 128, 3),     #(8x8)
#                     nn.MaxPool2d(2),            #(4x4)
#                     nn.ReLU(),
#                     nn.Conv2d(128, 256, 4),  # (1x1)
#                     nn.ReLU(),
#         )
#
#         self.linear = nn.Linear(self.endLayerSize,2121)
#
#     def forward(self, img):
#         x = self.conv(img)
#         x = x.view(self.inputSize,self.endLayerSize)
#         x = self.linear(x)
#         return x
#
# #v 0.4.1
# #desc: the analizer used for training
# class AnalizerCNN(nn.Module):
#     def __init__(self,inputSize):
#         super(AnalizerCNN, self).__init__()
#         self.inputSize = inputSize
#         self.endLayerSize = 256
#
#         #TODO:
#         #make Kernelsize variable and dependend on inputSize
#         #also add maxPools when variable is implemented
#         self.conv = nn.Sequential(
#                     nn.Conv2d(3, 64, 5),       #(21x21)
#                     nn.MaxPool2d(2),            #(10x10)?
#                     nn.ReLU(),
#                     nn.Conv2d(64, 128, 3),     #(8x8)
#                     nn.MaxPool2d(2),            #(4x4)
#                     nn.ReLU(),
#                     nn.Conv2d(128, 256, 4),  # (1x1)
#                     nn.ReLU(),
#         )
#
#         self.linear = nn.Linear(self.endLayerSize,2121)
#         # self.linear02 = nn.Linear(self.endLayerSize, 2121) #possible addition
#
#     def forward(self, img):
#         x = self.conv(img)
#         x = x.view(self.inputSize,self.endLayerSize)
#         x = self.linear(x)
#         return x





