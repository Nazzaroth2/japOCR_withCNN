from PIL import Image, ImageDraw, ImageFont
from lib import trainDataGenerationLib as dataLib
from torchvision.transforms import ColorJitter
import random
import torch
import torch.nn as nn



# filepathColor = 'trainingData\\kanjiUnicodeList\\colorSetting.csv'
# filepathUnicode = 'trainingData\\kanjiUnicodeList\\unicodeList.csv'
#
# dataGeneratorObj = dataLib.unicodeColorGenerator(filepathUnicode,filepathColor)
#
# dataGenerator = dataGeneratorObj.draw("togalite-regular.otf", 50, 0)
#
# Transformer = ColorJitter(0,0,0,(0.0,0.5))
#
# imgList = []
# for i in range(3000):
#     imgList.append(dataGeneratorObj.toTensorAndJitterConversion(next(dataGenerator),(0.1,0.8),0,0,0))
#
# for i in range(5):
#     randInt = random.randint(500,1000)
#     imgList[randInt].show()


loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)

print(input)
print()
print(target)
print()
print(output)






