from lib import trainDataGenerationLib as dataLib
from lib import trainingEvaluationLib as evalLib
from torchvision.transforms import ColorJitter, ToPILImage
import random
from time import time
from itertools import islice
import torch
import torch.nn as nn
import torchvision
import os


# if __name__ == '__main__':
#     startTime = time()
#
#     filepathColor = 'trainingData\\kanjiUnicodeList\\colorSetting.csv'
#     filepathUnicode = 'trainingData\\kanjiUnicodeList\\unicodeList.csv'
#     fonts = ["togalite-regular.otf",]
#     batchLength = 128
#
#     # randomSettings = [[0,0,0,0],[(0.1,0.8),0,0,0],[0,(0.1,0.8),0,0],[0,0,(0.1,0.8),0],[0,0,0,(-0.5,0.5)]]
#
#
#     transformer = torchvision.transforms.Compose([
#         # torchvision.transforms.RandomAffine((-20,20))
#         torchvision.transforms.RandomAffine(0,translate=(0.1,0.1))
#     ])
#
#     trainingSet = torchvision.datasets.ImageFolder("trainDataRoot",transformer)
#     # print(len(trainingSet))
#     for i in range(5):
#         randNum = random.randrange(0,4000)
#         print(trainingSet[randNum])








    # trainLoader = torch.utils.data.DataLoader(trainingSet, batchLength, shuffle=True, num_workers=5)

    # #trainingLoop
    # for i, data in enumerate(trainLoader,0):
    #     inputs, lables = data
    #
    #     for img in inputs:
    #         img.show()



    # endTime = time() - startTime
    #
    #
    # print("We took {} seconds to train, that is {} mins.".format(format(endTime,".3f"),
    #                                                              format(endTime/60,".3f")))



testNum = (0.13443295,8723098,45193845)

# print("{}".format(format(testNum,".3f")))


with open("logfileEvaluation.txt","w",encoding="utf-8") as f:
    f.write(str(testNum))















# #make trainPics and save them on disk
# filepathColor = 'trainingData\\kanjiUnicodeList\\colorSetting02.csv'
# filepathUnicode = 'trainingData\\kanjiUnicodeList\\unicodeList.csv'
# #to find windows fonts filenames go to:
# #registry->local_machine_software_microsoft_windowsNT_CurrentVersion_Fonts
# fonts = ["GenShinGothic-Normal.ttf","K Gothic.ttf","komorebi-gothic.ttf",
#          "NanigoSquare-Regular.ttf","OtsutomeFont_Ver3.ttf","Ronde-B_square.otf","Snap_P_ver1.otf",
#          "YuGothR.ttc","yumin.ttf","togalite-regular.otf","UDDigiKyokashoN-R.ttc",
#          "msgothic.ttc","msmincho.ttc","851MkPOP_1.ttf","Zomzi.TTF","Caramel_condenced_Ver1.00.otf",
#
#          ]
# rootPath = os.path.join('trainingData','trainPics05')
# for img,classValue in evalLib.makeMiniBatch(filepathUnicode, filepathColor, fonts, 25):
#     # evalLib.savePics(rootPath,img,classValue)
#     print(classValue)
