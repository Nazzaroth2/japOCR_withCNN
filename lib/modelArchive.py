"""
author: Viktor Cerny
created: 29-05-19
last edit: 29-05-19
desc: gravejard for obsolete model architecture versions, so main file stays clean
"""










# #v 0.3.4
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
#                     nn.Conv2d(3, 64, 5),       #(12x12)
#                     nn.MaxPool2d(2),            #(6x6)
#                     nn.ReLU(),
#                     nn.Conv2d(64, 128, 3),     #(4x4)
#                     nn.MaxPool2d(2),            #(2x2)
#                     nn.ReLU(),
#                     nn.Conv2d(128, 256, 2),  # (1x1)
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
# #v 0.3.4
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
#                     nn.Conv2d(3, 64, 5),       #(12x12)
#                     nn.MaxPool2d(2),            #(6x6)
#                     nn.ReLU(),
#                     nn.Conv2d(64, 128, 3),     #(4x4)
#                     nn.MaxPool2d(2),            #(2x2)
#                     nn.ReLU(),
#                     nn.Conv2d(128, 256, 2),  # (1x1)
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




# #v 0.3.2
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
#             nn.Conv2d(3, 128, 3),       # (14x14)
#             nn.MaxPool2d(2),            # (7x7)
#             nn.ReLU(),
#             nn.Conv2d(128, 256, 3),     # (5x5)
#             nn.ReLU(),
#             nn.Conv2d(256, 512, 3),     # (3x3)
#             nn.ReLU(),
#             nn.Conv2d(512, 1024, 3),    # (1x1)
#             nn.ReLU(),
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
# #v 0.3.3
# #desc: the analizer used for training
# class AnalizerCNN(nn.Module):
#     def __init__(self,inputSize):
#         super(AnalizerCNN, self).__init__()
#         self.inputSize = inputSize
#         self.endLayerSize = 64
#
#         #TODO:
#         #make Kernelsize variable and dependend on inputSize
#         #also add maxPools when variable is implemented
#         self.conv = nn.Sequential(
#                     nn.Conv2d(3, 64, 7),       #(10x10)
#                     nn.ReLU(),
#                     nn.Conv2d(64, 128, 5),     #(6x6)
#                     nn.ReLU(),
#                     nn.Conv2d(128, 64, 3),  # (4x4)
#                     nn.ReLU(),
#                     nn.Conv2d(64, 128, 3),  # (2x2)
#                     nn.ReLU(),
#                     nn.Conv2d(128, 64, 2),  # (1x1)
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




# #v 0.3.2
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
#             nn.Conv2d(3, 128, 3),       # (14x14)
#             nn.MaxPool2d(2),            # (7x7)
#             nn.ReLU(),
#             nn.Conv2d(128, 256, 3),     # (5x5)
#             nn.ReLU(),
#             nn.Conv2d(256, 512, 3),     # (3x3)
#             nn.ReLU(),
#             nn.Conv2d(512, 1024, 3),    # (1x1)
#             nn.ReLU(),
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
# #v 0.3.2
# #desc: the analizer used for training
# class AnalizerCNN(nn.Module):
#     def __init__(self,inputSize):
#         super(AnalizerCNN, self).__init__()
#         self.inputSize = inputSize
#         self.endLayerSize = 64
#
#         #TODO:
#         #make Kernelsize variable and dependend on inputSize
#         #also add maxPools when variable is implemented
#         self.conv = nn.Sequential(
#                     nn.Conv2d(3, 64, 3),       #(14x14)
#                     nn.ReLU(),
#                     nn.Conv2d(64, 128, 3),     #(12x12)
#                     nn.ReLU(),
#                     nn.Conv2d(128, 128, 3),     #(10x10)
#                     nn.ReLU(),
#                     nn.Conv2d(128, 64, 3),    #(8x8)
#                     nn.ReLU(),
#                     nn.Conv2d(64, 64, 3),  # (6x6)
#                     nn.ReLU(),
#                     nn.Conv2d(64, 128, 3),  # (4x4)
#                     nn.ReLU(),
#                     nn.Conv2d(128, 128, 3),  # (2x2)
#                     nn.ReLU(),
#                     nn.Conv2d(128, 64, 2),  # (1x1)
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




# #v 0.3.1
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
#             nn.Conv2d(3, 128, 3),       # (14x14)
#             nn.MaxPool2d(2),            # (7x7)
#             nn.ReLU(),
#             nn.Conv2d(128, 256, 3),     # (5x5)
#             nn.ReLU(),
#             nn.Conv2d(256, 512, 3),     # (3x3)
#             nn.ReLU(),
#             nn.Conv2d(512, 1024, 3),    # (1x1)
#             nn.ReLU(),
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
# #v 0.3.1
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
#                     nn.Conv2d(3, 512, 3),       #(14x14)
#                     nn.MaxPool2d(2),            #(7x7)
#                     nn.ReLU(),
#                     nn.Conv2d(512, 256, 3),     #(5x5)
#                     nn.ReLU(),
#                     nn.Conv2d(256, 128, 3),     #(3x3)
#                     nn.ReLU(),
#                     nn.Conv2d(128, 256, 3),    #(1x1)
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




# #v 0.3.0
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
#             nn.Conv2d(3, 128, 3),       # (14x14)
#             nn.MaxPool2d(2),            # (7x7)
#             nn.ReLU(),
#             nn.Conv2d(128, 256, 3),     # (5x5)
#             nn.ReLU(),
#             nn.Conv2d(256, 512, 3),     # (3x3)
#             nn.ReLU(),
#             nn.Conv2d(512, 1024, 3),    # (1x1)
#             nn.ReLU(),
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
# #v 0.3.0
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
#                     nn.Conv2d(3, 128, 3),       #(14x14)
#                     nn.MaxPool2d(2),            #(7x7)
#                     nn.ReLU(),
#                     nn.Conv2d(128, 256, 3),     #(5x5)
#                     nn.ReLU(),
#                     nn.Conv2d(256, 512, 3),     #(3x3)
#                     nn.ReLU(),
#                     nn.Conv2d(512, 1024, 3),    #(1x1)
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




# #v 0.2.0
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
#                     nn.Conv2d(3, 128, 5),
#                     nn.ReLU(),
#                     nn.Conv2d(128,256,5),
#                     nn.ReLU(),
#                     nn.Conv2d(256, self.endLayerSize, 8),
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
# #v 0.2.0
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
#                     nn.Conv2d(3, 128, 5),
#                     nn.ReLU(),
#                     nn.Conv2d(128,256,5),
#                     nn.Dropout(),
#                     nn.ReLU(),
#                     nn.Conv2d(256, self.endLayerSize, 8),
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

# #v 0.1.4
# #desc: the analizer used for training
# class AnalizerCNN(nn.Module):
#     def __init__(self,inputSize):
#         super(AnalizerCNN, self).__init__()
#         self.inputSize = inputSize
#         self.endLayerSize = 512
#
#
#         #TODO:
#         #make Kernelsize variable and dependend on inputSize
#         #also add maxPools when variable is implemented
#         self.conv = nn.Sequential(
#                     nn.Conv2d(3, 128, 16),
#                     nn.ReLU(),
#                     nn.Conv2d(128,256,1),
#                     nn.Dropout(),
#                     nn.ReLU(),
#                     nn.Conv2d(256, self.endLayerSize, 1),
#                     nn.ReLU(),
#         )
#
#         self.linear = nn.Linear(self.endLayerSize,2121)
#
#
#
#     def forward(self, img):
#         x = self.conv(img)
#         x = x.view(self.inputSize,self.endLayerSize)
#         x = self.linear(x)
#         return x

# #v 0.1.3
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
#                     nn.Conv2d(3, 128, 16),
#                     nn.ReLU(),
#                     nn.Conv2d(128,256,1),
#                     nn.ReLU(),
#                     nn.Conv2d(128,256,1),
#                     nn.Dropout(),
#                     nn.ReLU(),
#                     nn.Conv2d(256,256,1),
#                     nn.ReLU(),
#                     nn.Conv2d(256, self.endLayerSize, 1),
#                     nn.ReLU(),
#         )
#
#         self.linear = nn.Linear(self.endLayerSize,2121)
#
#
#     def forward(self, img):
#         x = self.conv(img)
#         x = x.view(self.inputSize,self.endLayerSize)
#         x = self.linear(x)
#         return x

# #v 0.1.2
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
#                     nn.Conv2d(3, 32, 16),
#                     nn.ReLU(),
#                     nn.Conv2d(32,64,1),
#                     nn.ReLU(),
#                     nn.Conv2d(64,128,1),
#                     nn.Dropout(),
#                     nn.ReLU(),
#                     nn.Conv2d(128,256,1),
#                     nn.ReLU(),
#                     nn.Conv2d(256, self.endLayerSize, 1),
#                     nn.ReLU(),
#         )
#
#         self.linear = nn.Linear(self.endLayerSize,2121)
#
#
#     def forward(self, img):
#         x = self.conv(img)
#         x = x.view(self.inputSize,self.endLayerSize)
#         x = self.linear(x)
#         return x

# #v 0.1.1
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
#                     nn.Conv2d(3, 32, 16),
#                     nn.ReLU(),
#                     nn.Conv2d(32,64,1),
#                     nn.ReLU(),
#                     nn.Conv2d(64,128,1),
#                     nn.Dropout(),
#                     nn.ReLU(),
#                     nn.Conv2d(128,256,1),
#                     nn.ReLU(),
#                     nn.Conv2d(256, 512, 1),
#                     nn.ReLU(),
#                     nn.Conv2d(512, 512, 1),
#                     nn.Dropout(),
#                     nn.ReLU(),
#                     nn.Conv2d(512, self.endLayerSize, 1),
#                     nn.ReLU(),
#         )
#
#         self.linear = nn.Linear(self.endLayerSize,2121)
#
#
#     def forward(self, img):
#         x = self.conv(img)
#         x = x.view(self.inputSize,self.endLayerSize)
#         x = self.linear(x)
#         return x



# #version 0.1.0
# #desc: the analizer used for training
# class AnalizerCNN(nn.Module):
#     def __init__(self,inputSize):
#         super(AnalizerCNN, self).__init__()
#         self.inputSize = inputSize
#
#         #TODO:
#         #make Kernelsize variable and dependend on inputSize
#         #also add maxPools when variable is implemented
#         self.conv = nn.Sequential(
#                     nn.Conv2d(3,64,16),
#                     nn.ReLU(),
#                     nn.Conv2d(64,128,1),
#                     nn.Dropout(),
#                     nn.ReLU(),
#                     nn.Conv2d(128,256,1),
#                     nn.ReLU(),
#         )
#
#         self.linear = nn.Linear(256,2121)
#
#
#     def forward(self, img):
#         x = self.conv(img)
#         x = x.view(self.inputSize,256)
#         x = self.linear(x)
#         return x




# #desc: the analizer used for evaluation
# class AnalizerCNN02(nn.Module):
#     def __init__(self, inputBatchSize, winWidth, winHeight):
#         super(AnalizerCNN02, self).__init__()
#         self.inputBatchSize = inputBatchSize
#         self.winWidth = winWidth
#         self.winHeight = winHeight
#         self.kanjiRecognitionMap = {}
#         self.moveCounter = 0
#
#         #TODO:
#         #make Kernelsize variable and dependend on inputSize
#         #also add maxPools when variable is implemented
#         self.conv = nn.Sequential(
#                     nn.Conv2d(3,64,16),
#                     nn.ReLU(),
#                     nn.Conv2d(64,128,1),
#                     nn.Dropout(),
#                     nn.ReLU(),
#                     nn.Conv2d(128,256,1),
#                     nn.ReLU(),
#         )
#
#         self.linear = nn.Linear(256,2121)
#
#
#     def forward(self, img):
#         x = self.conv(img)
#         x = x.view(self.inputBatchSize, 256)
#         x = self.linear(x)
#
#         key = self.__keyCalculation()
#
#         self.kanjiRecognitionMap[key] = x
#
#
#
#
#
#
#         return self.kanjiRecognitionMap