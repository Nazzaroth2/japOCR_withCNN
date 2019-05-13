"""
author: Viktor Cerny
created: 13-05-19
last edit: 13-05-19
desc: the evaluation loop over a testimage. looping over the image
from the outside.
"""


from lib import trainingEvaluationLib as trainLib
from lib import model
import os
import torch

filepath = "trainingData\\Combo1_24bit.bmp"
height = width = 16

evalBatch = trainLib.makeEvalutionBatch(trainLib.windowCrop(filepath,height,width))

loadPath = "trainedNets\\analizerCNN_Adam_Cross_001.pt"
batchLength = len(evalBatch)

#standard training objects creation
AnalizerCNN = model.AnalizerCNN(batchLength)

#load trained Net if trained Net exists
if os.path.isfile(loadPath):
    AnalizerCNN.load_state_dict(torch.load(loadPath))
else:
    pass

analizedList = AnalizerCNN(evalBatch)


for i in range(5):
    print((analizedList[i] == torch.max(analizedList[i])).nonzero().item())
    print(torch.max(analizedList[i]).item())

