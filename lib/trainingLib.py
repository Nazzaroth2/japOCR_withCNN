"""
author: Viktor Cerny
created: 06-05-19
last edit: 06-05-19
desc: holds all functions that are neccesary for training and evaluation
"""

from lib import trainDataGenerationLib as genLib
from torch import unsqueeze,cat
import torch


#desc: encode unicodeChars into onehot representation
def unicodeOneHot(unicodeChar, unicodeList):
    oneHotList = [0] * 2120

    position = unicodeList.index(unicodeChar)

    oneHotList.insert(position, 1)

    return oneHotList

#desc: deals with creating a MiniBatch for training
def makeMiniBatch(filepathUnicode, filepathColor,font,size,batchLength):
    #create generator with specific font and size
    trainGenObj = genLib.unicodeColorGenerator(filepathUnicode, filepathColor)
    trainGen = trainGenObj.draw(font, size)

    answerBatch = []

    #initialize both Batches with 1 value each
    trainImg, unicodeChar = next(trainGen)
    imgBatch = unsqueeze(trainGenObj.toTensorConversion(trainImg), 0)
    #append the position index of used unicodeChar
    answerBatch.append(trainGenObj.unicodeList.index(unicodeChar))

    #add batchLength-1 amount of values to each batch
    for i in range(batchLength-1):
        trainImg, unicodeChar = next(trainGen)

        imgT = trainGenObj.toTensorConversion(trainImg)
        imgT = unsqueeze(imgT, 0)
        #cat combines Tensors in 0-dim [0[1[2[3]]]]
        imgBatch = cat((imgBatch, imgT), 0)

        answerBatch.append(trainGenObj.unicodeList.index(unicodeChar))

    answerTensor = torch.LongTensor(answerBatch)


    return imgBatch, answerTensor

