"""
author: Viktor Cerny
created: 08-04-19
last edit: 15-04-19
desc: library for functions that deal with image loading and pixel data extraction etc.
"""

import win32gui as gui
import win32ui
import win32con
from time import sleep
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor


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

        sleep(0.5)

#desc: calculates the position of targeted window given a handle
def getTargetWindowPosition(winHandle):
    targetPosition = gui.GetWindowRect(winHandle)

    return targetPosition

#desc: wrapper to the get targetWindowPosition Call
def returnTargetWindowPosition():
    return getTargetWindowPosition(captureTargetWindow())

#desc: returns a tuple of 4channel pixel values of target window space
#order is BGR-Alpha
def returnRawTarget():
    #get window space
    taskWindowValues = returnTargetWindowPosition()

    taskX = taskWindowValues[0]
    taskY = taskWindowValues[1]
    taskW = taskWindowValues[2] - taskX
    taskH = taskWindowValues[3] - taskY

    # get the desktop
    hwin = gui.GetDesktopWindow()
    hwindc = gui.GetWindowDC(hwin)

    # do other necessery stuff (should figure out eventually what all this stuff does)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, taskW, taskH)
    memdc.SelectObject(bmp)

    # copy desktop region into bmp object
    memdc.BitBlt((0, 0), (taskW, taskH), srcdc, (taskX, taskY), win32con.SRCCOPY)

    return bmp.GetInfo(), bmp.GetBitmapBits()

#desc: give the image out in a torch Tensor format to be fed right into the net
#conversion-line: raw -> np.array -> pil.Image -> torch.Tensor
def convertRawToTensor():
    rawInfo, rawPixels = returnRawTarget()

    array = np.asarray(rawPixels, dtype=np.uint8)

    pil_im = Image.frombuffer('RGB', (rawInfo['bmWidth'], rawInfo['bmHeight']),array, 'raw', 'BGRX', 0, 1)


    tensor = ToTensor().__call__(pic=pil_im)

    return tensor
