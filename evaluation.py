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
import torchvision

def evaluation(netPath,rootPath):
    Transformer = torchvision.transforms.ToTensor()
    batchSize = 15

    # testTrainingset
    testSet = torchvision.datasets.ImageFolder(rootPath, Transformer, trainLib.testTargetTransformBonW)
    testLoader = torch.utils.data.DataLoader(testSet, batchSize, shuffle=False, num_workers=1)

    # pretrained net
    loadPath = netPath

    # standard training objects creation
    AnalizerCNN = model.AnalizerCNN(8,11,2121,64,4)

    # load trained Net if trained Net exists
    if os.path.isfile(loadPath):
        AnalizerCNN.load_state_dict(torch.load(loadPath))
    else:
        raise FileNotFoundError('could not load pretrained net')

    errorGuess = {}
    rightGuess = {}
    errorSum = 0
    for data in testLoader:
        inputImages, classValues = data

        output = AnalizerCNN(inputImages)

        for pos, guess in enumerate(output):
            guessClass = (guess == torch.max(guess)).nonzero().item()
            if guessClass != classValues[pos].item():
                errorGuess[pos] = (guessClass,guess[guessClass].item(),classValues[pos].item(),
                                   guess[classValues[pos].item()].item())
                errorSum += 1
            else:
                errorGuess.append(0)
                rightGuess[guessClass] = guess[guessClass]
                rightGuess[classValues[pos].item()] = guess[classValues[pos].item()]



    print("The test showed:")
    print("The Net had {} false guesses.".format(errorSum))
    print("That is {} percent of the whole testSet".format((errorSum / len(testSet)) * 100))
    print("The wrongly guesses Images are at these positions:")
    for k in errorGuess:
        print("{}\t{}".format(str(errorGuess[k][0]),format(errorGuess[k][1],".2f")))
        print("{}\t{}".format(str(errorGuess[k][2]),format(errorGuess[k][3],".2f")))
        print()

    with open("logfileEvaluation.txt","w",encoding="utf-8") as f:
        f.write(str(errorGuess))
    # print("The rightly guesses Images are at these positions:")
    # print(rightGuess)




# if __name__ == '__main__':
#     from lib import trainingEvaluationLib as trainLib
#     from lib import model
#     import os
#     import torch
#     import torchvision
#
#     rootPath = "testDataRootBonW"
#     Transformer = torchvision.transforms.ToTensor()
#     batchLength = 1
#
#     #testTrainingset
#     testSet = torchvision.datasets.ImageFolder(rootPath,Transformer,trainLib.testTargetTransformBonW)
#     testLoader = torch.utils.data.DataLoader(testSet,batchLength,shuffle=True, num_workers=5)
#
#
#
#     #pretrained net
#     loadPath = os.path.join("trainedNets","analizerCNNv0.3.4_Adam_Cross_006.pt")
#
#     #standard training objects creation
#     AnalizerCNN = model.AnalizerCNNEval(batchLength)
#
#     #load trained Net if trained Net exists
#     if os.path.isfile(loadPath):
#         AnalizerCNN.load_state_dict(torch.load(loadPath))
#     else:
#         raise FileNotFoundError('could not load pretrained net')
#
#
#     errorPos = []
#     errorSum = 0
#     for data in testLoader:
#         inputImages, classValues = data
#
#         output = AnalizerCNN(inputImages)
#
#         for pos,guess in enumerate(output):
#             guessClass = (guess == torch.max(guess)).nonzero().item()
#             if guessClass != classValues[pos].item():
#                 errorPos.append((guessClass,classValues[pos].item()))
#                 errorSum += 1
#             else:
#                 errorPos.append(0)
#
#
#     print("The test showed:")
#     print("The Net had {} false guesses.".format(errorSum))
#     print("That is {} percent of the whole testSet".format((errorSum/37)*100))
#     print("The wrongly guesses Images are at these positions:")
#     print(errorPos)









# #cutting testpic into crops
# from lib import trainingEvaluationLib as trainLib
# filepath = "trainingData\\LN_BonW_Text_Line2_24bit.bmp"
# height = width = 16
#
# #make/save evalPics
# for img in trainLib.windowCrop(filepath,height,width):
#     trainLib.savePics("testDataRootBonW_FULL",img,-1)


    # for i in range(5):
    #     index = (analizedList[i] == torch.max(analizedList[i])).nonzero().item()
    #     indexValue = torch.max(analizedList[i]).item()
    #     indexUnicode = unicodeList[index]
    #     unicodeChar = chr(int(indexUnicode,16))
    #     print("The ",str(i),"piece is guessed as:")
    #     print(index)
    #     print(indexValue)
    #     print(indexUnicode)
    #     print(unicodeChar)
    #     print()


