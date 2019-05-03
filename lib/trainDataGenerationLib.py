"""
author: Viktor Cerny
created: 25-04-19
last edit: 30-04-19
desc: library for functions that deal with training data generation from
 unicode and fonts and later transformation of those pics
"""

from csv import reader
from textwrap import wrap
from PIL import Image, ImageFont, ImageDraw
from torchvision import transforms
from time import sleep

class unicodeColorGenerator:
    def __init__(self,filepathUnicode,filepathColor):
        self.unicodeList = self.__readUnicodeCSV(filepathUnicode)
        self.colorSettings = self.__readColorCSV(filepathColor)


    #desc: reads japanese Char Unicodes from file and stores in a list
    def __readUnicodeCSV(self,filepathUnicode):
        unicodeList = []

        try:
            with open(filepathUnicode, 'r') as f:
                fileReader = reader(f, delimiter=',')
                for row in fileReader:
                    unicodeList.append(row[0])
        except FileNotFoundError:
            print("You didn't give the right Unicode FilePath")
            sleep(0.1)
            raise


        return unicodeList


    #desc: reads the Color Settings from file and stores them in this structure:
    #list(backg-color=['XXX','XXX','XXX']; foreg-color=['#XXXXXX'])
    def __readColorCSV(self, filepathColor):
        colorSettings = []

        try:
            with open(filepathColor, 'r') as f:
                sep = '#'
                fileReader = reader(f, delimiter=',')
                for pos,row in enumerate(fileReader):
                    colorSettings.append(row[0].split(sep))
                    #backgroundColor
                    colorSettings[pos][0] = wrap(colorSettings[pos][0],3)
                    #charColor
                    colorSettings[pos][1] = sep + colorSettings[pos][1]
        except FileNotFoundError:
            print("You didn't give the right ColorSetting FilePath")
            sleep(0.1)
            raise

        return colorSettings

    #desc: create font object
    def __createFontObj(self,font,size):
        return ImageFont.truetype(font, size, encoding="unic")

    #desc: setup the canvas to draw the char
    def __canvasSetup(self, fontObj, char, padding,backgroundColor):
        text_w, text_h = fontObj.getsize(char)

        canvas = Image.new('RGB', (text_w + padding, text_h + padding), backgroundColor)

        return canvas

    #desc: turn the unicode string from file into the corresponding character
    #becausce chr wants the int value of unicode we decode the 16base hex from file into int
    def __fileToUnicode(self,unicode):
        return chr(int(unicode,16))

    #desc: unpacks the Colorvalues in seperate variables
    def __colorUnpacker(self,color):
        backgroundColor = (int(color[0][0]), int(color[0][1]), int(color[0][2]))
        charColor = color[1]

        return backgroundColor, charColor

    def toTensorConversion(self,img):
        transformer = transforms.ToTensor()
        return transformer(img)

    def toTensorAndJitterConversion(self,img,brightness, contrast, saturation, hue):
        transformer = transforms.Compose([
            transforms.ColorJitter(brightness,contrast,saturation,hue),
            transforms.ToTensor(),
        ])

        return transformer(img)


    #desc: creates a generator for a specific font-size-padding combination that
    #returns PIL-Images for every color-Character Combination
    def draw(self, font, size, padding):

        fontObj = self.__createFontObj(font, size)

        for color in self.colorSettings:
            backgroundColor, charColor = self.__colorUnpacker(color)

            for unicode in self.unicodeList:
                unicode = self.__fileToUnicode(unicode)

                canvas = self.__canvasSetup(fontObj, unicode, padding, backgroundColor)

                drawObj = ImageDraw.Draw(canvas)
                drawObj.text((0, 0), unicode, charColor, fontObj)

                yield canvas
