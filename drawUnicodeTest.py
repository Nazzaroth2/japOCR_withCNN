from PIL import Image, ImageDraw, ImageFont
from lib import trainDataGenerationLib as dataLib



# fileCode = '5019'
# hexCode = int(fileCode,16)
# unicode_char = chr(int(hexCode))
#
# myFont = ImageFont.truetype("851MkPOP.ttf", 30, encoding="unic")
#
# #size of the text
# text_width, text_height = myFont.getsize(unicode_char)
# print(text_width," ",text_height)
#
# #create a blank Canvas
# canvas = Image.new('RGB', (text_width + 10, text_height + 10), (10,50,140))
#
# #draw the text onto the canvas, using black as color
# draw = ImageDraw.Draw(canvas)
# draw.text((5,5), unicode_char, font=myFont, fill= "#ffffff")
#
# #show the image
# canvas.show()
#
filepathColor = 'trainingData\\kanjiUnicodeList\\colorSetting.csv'
filepathUnicode = 'trainingData\\kanjiUnicodeList\\unicodeList.csv'

dataGeneratorObj = dataLib.unicodeColorGenerator(filepathUnicode,filepathColor)

dataGenerator = dataGeneratorObj.draw("togalite-regular.otf", 30, 0)

imgList = []
for i in range(500):
    imgList.append(next(dataGenerator))

# import random
# randNum = random.randint(0,500)
# print(randNum)

imgList[170].show()





