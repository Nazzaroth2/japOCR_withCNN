"""
author: Viktor Cerny
created: 14-05-19
last edit: 14-05-19
desc: functions and classes that have a variety of usecases.
"""

from csv import reader
from textwrap import wrap

class fileInputSystem:
    # desc: reads japanese Char Unicodes from file and stores in a list
    @classmethod
    def readUnicodeCSV(cls,filepathUnicode):
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

    # desc: reads the Color Settings from file and stores them in this structure:
    # list(backg-color=['XXX','XXX','XXX']; foreg-color=['#XXXXXX'])
    @classmethod
    def readColorCSV(cls, filepathColor):
        colorSettings = []

        try:
            with open(filepathColor, 'r') as f:
                sep = '#'
                fileReader = reader(f, delimiter=',')
                for pos, row in enumerate(fileReader):
                    colorSettings.append(row[0].split(sep))
                    # backgroundColor
                    colorSettings[pos][0] = wrap(colorSettings[pos][0], 3)
                    # charColor
                    colorSettings[pos][1] = sep + colorSettings[pos][1]
        except FileNotFoundError:
            print("You didn't give the right ColorSetting FilePath")
            sleep(0.1)
            raise

        return colorSettings