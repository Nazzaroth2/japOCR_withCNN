"""
author: Viktor Cerny
created: 06-05-19
last edit: 06-05-19
desc: holds all functions that are neccesary for training and evaluation
"""

from lib import trainDataGenerationLib as genLib
from torch import unsqueeze,cat
from PIL import Image
from torchvision.transforms import ToTensor
import csv
import os


#desc: deals with creating pic list for saving
def makeMiniBatch(filepathUnicode, filepathColor,fonts,size):

    for font in fonts:
        trainGenObj = genLib.unicodeColorGenerator(filepathUnicode, filepathColor)
        trainGen = trainGenObj.draw(font, size)

        #loop through all generator values
        for trainImg, unicodeChar in trainGen:

            classValue = trainGenObj.unicodeList.index(unicodeChar)

            yield trainImg, classValue


#saves a single training pic
def savePics(rootPath,pic,classValue,counter=0):
    classPath = os.path.join(rootPath,str(classValue))
    try:
        os.makedirs(classPath)
    except OSError:
        pass


    filePath = os.path.join(rootPath, str(classValue), "pic"+str(counter)+".bmp")
    while(os.path.isfile(filePath)):
        counter += 1
        filePath = os.path.join(rootPath, str(classValue), "pic"+str(counter)+".bmp")

    try:
        pic.save(filePath)
    except Exception as e:
        print("File",filePath,"could not be saved.")
        print(e)





#TODO
#combine these functions with torchvison.dataset.ImageFolder
#to have a testset creator(on hold cause i have to sort crops by hand right now)
#desc: cuts big img into given size with a given step
def windowCrop(filePath,height,width,heightStep=3,widthStep=3):
    im = Image.open(filePath)
    imgwidth, imgheight = im.size
    for i in range(0,(imgheight-height),heightStep):
        for j in range(0,(imgwidth-width),widthStep):
            box = (j, i, j+width, i+height)
            yield im.crop(box)


#desc: Transforms the target to actuall kanji classes for pic "Combo1_24bit.bmp"
def testTargetTransform(orgTarget):
    transFormDict = {0:5,1:11,2:22,3:28,4:30,5:42,6:45,
                     7:65,8:76,9:82,10:175,11:217,12:361,13:594,
                     14:612,15:798,16:825,17:1332,18:1393}

    return transFormDict[orgTarget]

#desc: Transforms the target to actuall kanji classes for pic "Combo1_24bit.bmp"
def testTargetTransformBonW(orgTarget):
    transFormDict = {0:34,1:37,2:42,3:47,4:71,5:72,6:74,
                     7:75,8:183,9:186,10:254,11:379,12:482,13:785,
                     14:1351,15:1751}

    return transFormDict[orgTarget]



