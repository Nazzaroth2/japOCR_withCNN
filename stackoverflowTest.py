import win32gui, win32ui, win32con, win32api

game = win32gui.FindWindow(None, "アールココ-片翼の愛玩姫-2.0")
taskWindowValues = win32gui.GetWindowRect(game)
offset = 10
taskX = taskWindowValues[0] + offset
taskY = taskWindowValues[1] + offset
taskW = taskWindowValues[2] - taskX - offset
taskH = taskWindowValues[3] - taskY - offset

hwin = win32gui.GetDesktopWindow()


# print(width)
# print(height)
# print(left)
# print(top)
# print()
#
# print(taskWindowValues[0])
# print(taskWindowValues[1])
# print(taskWindowValues[2])
# print(taskWindowValues[3])


hwindc = win32gui.GetWindowDC(hwin)
srcdc = win32ui.CreateDCFromHandle(hwindc)
memdc = srcdc.CreateCompatibleDC()
bmp = win32ui.CreateBitmap()
bmp.CreateCompatibleBitmap(srcdc, taskW, taskH)
memdc.SelectObject(bmp)

memdc.BitBlt((0, 0), (taskW, taskH), srcdc, (taskX, taskY), win32con.SRCCOPY)
bmp.SaveBitmapFile(memdc, 'screenshot.bmp')