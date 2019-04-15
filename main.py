"""
author: Viktor Cerny
created: 08-04-19
last edit: 08-04-19
desc: main file for the project. starts up the program
"""

import win32gui as gui

import time

import imageInputLib as imgLib

actHandle = imgLib.captureTargetWindow()

print(actHandle)


