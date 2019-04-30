"""
author: Viktor Cerny
created: 08-04-19
last edit: 08-04-19
desc: main file for the project. starts up the program
"""

from lib import imageInputLib as imgLib

tensor = imgLib.convertRawToTensor()

print(tensor.size())

