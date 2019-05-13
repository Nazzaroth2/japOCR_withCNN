"""
author: Viktor Cerny
created: 03-05-19
last edit: 03-05-19
desc: all neural-network-models (classes) are found in this file
"""

import torch.nn as nn

#desc: the analizer used for training
class AnalizerCNN(nn.Module):
    def __init__(self,inputSize):
        super(AnalizerCNN, self).__init__()
        self.inputSize = inputSize

        #TODO:
        #make Kernelsize variable and dependend on inputSize
        #also add maxPools when variable is implemented
        self.conv = nn.Sequential(
                    nn.Conv2d(3,64,16),
                    nn.ReLU(),
                    nn.Conv2d(64,128,1),
                    nn.Dropout(),
                    nn.ReLU(),
                    nn.Conv2d(128,256,1),
                    nn.ReLU(),
        )

        self.linear = nn.Linear(256,2121)


    def forward(self, img):
        x = self.conv(img)
        x = x.view(self.inputSize,256)
        x = self.linear(x)
        return x







#desc: the analizer used for evaluation
class AnalizerCNN02(nn.Module):
    def __init__(self, inputBatchSize, winWidth, winHeight):
        super(AnalizerCNN02, self).__init__()
        self.inputBatchSize = inputBatchSize
        self.winWidth = winWidth
        self.winHeight = winHeight
        self.kanjiRecognitionMap = {}
        self.moveCounter = 0

        #TODO:
        #make Kernelsize variable and dependend on inputSize
        #also add maxPools when variable is implemented
        self.conv = nn.Sequential(
                    nn.Conv2d(3,64,16),
                    nn.ReLU(),
                    nn.Conv2d(64,128,1),
                    nn.Dropout(),
                    nn.ReLU(),
                    nn.Conv2d(128,256,1),
                    nn.ReLU(),
        )

        self.linear = nn.Linear(256,2121)


    def forward(self, img):
        x = self.conv(img)
        x = x.view(self.inputBatchSize, 256)
        x = self.linear(x)

        key = self.__keyCalculation()

        self.kanjiRecognitionMap[key] = x






        return self.kanjiRecognitionMap