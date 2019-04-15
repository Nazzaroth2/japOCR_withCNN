"""
author: Viktor Cerny
created: 08-04-19
last edit: 15-04-19
desc: library for functions that deal with image loading and pixel data extraction etc.
"""

import win32gui as gui
from time import sleep

#desc: returns the handle of the active window
def getWindowHandle():
    window = gui.GetForegroundWindow()
    return window


#desc: waits until the target window is activated by user and returns handle of that window
def captureTargetWindow():
    orgHandle = getWindowHandle()
    print("please click the window you want to capture")
    while True:
        actHandle = getWindowHandle()
        if actHandle != orgHandle:
            # print can be used for later asking user if window was the right window
            # but we will just assume i clicked the right window for now
            # print("you have captured the window with the the handle {}".format(actHandle))
            return actHandle
            break

        sleep(1)