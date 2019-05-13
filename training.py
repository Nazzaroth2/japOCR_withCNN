"""
author: Viktor Cerny
created: 04-05-19
last edit: 06-05-19
desc: everything that has to do with training and trainingloop, but cant
be put in external lib
"""

# from lib import imageInputLib as imgLib
# from lib import trainDataGenerationLib as genLib

import torch.optim as optim
import torch.nn as nn
import torch

from time import time
import os

from lib import trainingEvaluationLib as trainLib
from lib import model

startTime = time()


#HYPERPARAMETER / FilePaths
loadPath = "trainedNets\\analizerCNN_Adam_Cross_001.pt"
filepathColor = 'trainingData\\kanjiUnicodeList\\colorSetting.csv'
filepathUnicode = 'trainingData\\kanjiUnicodeList\\unicodeList.csv'
font = "togalite-regular.otf"

batchLength = 2121
epoch = 100
LEARNING_RATE = 0.001


#standard training objects creation
AnalizerCNN = model.AnalizerCNN(batchLength)
optimizer = optim.Adam(AnalizerCNN.parameters(),lr=LEARNING_RATE)
lossFunction = nn.CrossEntropyLoss()

#load trained Net if trained Net exists
if os.path.isfile(loadPath):
    AnalizerCNN.load_state_dict(torch.load(loadPath))
else:
    pass



imgBatch, answerBatch = trainLib.makeMiniBatch(filepathUnicode,filepathColor,font,16,batchLength)

#gpu training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
AnalizerCNN.to(device)
imgBatch = imgBatch.to(device)
answerBatch = answerBatch.to(device)



#trainingsLoop
for i in range(epoch):

    output = AnalizerCNN(imgBatch)

    optimizer.zero_grad()

    loss = lossFunction(output,answerBatch)

    loss.backward()

    optimizer.step()

    if i % 500 == 0:
        print("Epoch:", i)
        print(loss.item())
        print()


endTime = time() - startTime


#eval
output = AnalizerCNN(imgBatch)
errors = []


for pos,i in enumerate(output):
    if (i == torch.max(i)).nonzero().item() != answerBatch[pos].item():
        errors.append(1)
    else:
        pass

limit = []
for i in output:
    limit.append(torch.max(i).item())



for i in range(5):
    print(output[i])
    print()
    print(limit[i])
    print()
    print()








# errorSum = sum(errors)
# errorPercent = errorSum / (batchLength / 100)
#
#
#
# print("End sum error is: {} and percentage is: {}".format(errorSum,format(errorPercent,".3f")))
# print("We took {} seconds to train, that is {} mins.".format(format(endTime,".3f"),
#                                                              format(endTime/60,".3f")))
#
# #saving Weights
# savePath = "trainedNets\\analizerCNN_Adam_Cross_001.pt"
# try:
#     torch.save(AnalizerCNN.state_dict(),savePath)
# except:
#     print("Save was not succesfull.")
# else:
#     print("Save was succesfull.")
#
#
#


