"""
author: Viktor Cerny
created: 08-04-19
last edit: 08-04-19
desc: main file for the project. starts up the program
"""

import win32gui as gui
from PIL import ImageGrab


git = gui.FindWindow(None, "japOCR (experimental) - Git Extensions")
game = gui.FindWindow(None, "アールココ-片翼の愛玩姫-2.0")

print(type(git))
print(git)
print(type(game))
print(game)

gitWindowValues = gui.GetWindowRect(git)
taskWindowValues = gui.GetWindowRect(game)

taskX = taskWindowValues[0]
taskY = taskWindowValues[1]
taskW = taskWindowValues[2] - taskX
taskH = taskWindowValues[3] - taskY

print("the x and y for taskmanager are {} and {}".format(taskX,taskY))
print("the window has {}px width and {}px height".format(taskW,taskH))

bbox = [taskWindowValues[0],taskWindowValues[1],
        taskWindowValues[2],taskWindowValues[3]]

print(bbox)

img = ImageGrab.grab(bbox)

print(type(img))

img.show()
