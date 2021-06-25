import cv2


def cv2_waitKey(window, delay):
    """Like cv2.waitKey, but also unblocks if window is closed"""
    waited = 0
    while waited < delay or delay == 0:
        key = cv2.waitKeyEx(10)
        waited += 1
        if key != -1:
            return key
        if cv2.getWindowProperty(window, 0) < 0:
            return -2
    return -1


def cv2_upscale(image, factor=2):
    dsize = (int(image.shape[1] * factor), int(image.shape[0] * factor))
    return cv2.resize(image, dsize=dsize, interpolation=cv2.INTER_NEAREST)
