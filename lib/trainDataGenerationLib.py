"""
author: Viktor Cerny
created: 25-04-19
last edit: 14-05-19
desc: library for functions that deal with training data generation from
 unicode and fonts and later transformation of those pics
"""


from PIL import Image, ImageFont, ImageDraw
from torchvision import transforms
from lib.generalLib import fileInputSystem



class unicodeColorGenerator:
    def __init__(self,filepathUnicode,filepathColor):
        self.unicodeList = fileInputSystem.readUnicodeCSV(filepathUnicode)
        self.colorSettings = fileInputSystem.readColorCSV(filepathColor)


    #desc: create font object
    def __createFontObj(self,font,size):
        return ImageFont.truetype(font, size, encoding="unic")

    #desc: setup the canvas to draw the char
    def __canvasSetup(self, fontObj, char, size,backgroundColor):
        canvas = Image.new('RGB', (size, size), backgroundColor)

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

    def jitterConversion(self,img,brightness, contrast, saturation, hue):
        transformer = transforms.Compose([
            transforms.ColorJitter(brightness,contrast,saturation,hue),
        ])

        return transformer(img)


    #desc: creates a generator for a specific font-size-padding combination that
    #returns PIL-Images for every color-Character Combination
    def draw(self, font, size):
        fontObj = self.__createFontObj(font, size)

        for color in self.colorSettings:
            backgroundColor, charColor = self.__colorUnpacker(color)

            for unicodeRaw in self.unicodeList:
                unicodeChar = self.__fileToUnicode(unicodeRaw)

                canvas = self.__canvasSetup(fontObj, unicodeChar, size, backgroundColor)

                drawObj = ImageDraw.Draw(canvas)
                drawObj.text((0, 0), unicodeChar, charColor, fontObj)

                yield canvas, unicodeRaw
