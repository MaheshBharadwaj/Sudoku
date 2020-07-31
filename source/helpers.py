import cv2
import numpy as np


def largest4SideContour(image):
    contours, h = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sorting contours in descending order based on the area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Looking at top 5 contours and checking if any of them are of size 4
    for cnt in contours[:min(5, len(contours))]:
        if len(approx(cnt)) == 4:
            return cnt
    return None


def approx(cnt):
    try:
        peri = cv2.arcLength(cnt, True)
        app = cv2.approxPolyDP(cnt, 0.01 * peri, True)
        return app
    except:
        return None


def get_rectangle_corners(cnt):
    pts = cnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point has the smallest sum whereas the
    # bottom-right has the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # compute the difference between the points -- the top-right
    # will have the minumum difference and the bottom-left will
    # have the maximum difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_perspective(rect, grid):
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(grid, M, (maxWidth, maxHeight))
    return cv2.resize(warp, (288, 288))


def getTopLine(image):
    for i, row in enumerate(image):
        if np.any(row):
            return i
    return None


def getBottomLine(image):
    for i in range(image.shape[0] - 1, -1, -1):
        if np.any(image[i]):
            return i
    return None


def getLeftLine(image):
    for i in range(image.shape[1]):
        if np.any(image[:, i]):
            return i
    return None


def getRightLine(image):
    for i in range(image.shape[1] - 1, -1, -1):
        if np.any(image[:, i]):
            return i
    return None


def rowShift(image, start, end, length):
    shifted = np.zeros(image.shape)
    if start + length < 0:
        length = -start
    elif end + length >= image.shape[0]:
        length = image.shape[0] - 1 - end

    for row in range(start, end + 1):
        shifted[row + length] = image[row]
    return shifted


def colShift(image, start, end, length):
    shifted = np.zeros(image.shape)
    if start + length < 0:
        length = -start
    elif end + length >= image.shape[1]:
        length = image.shape[1] - 1 - end

    for col in range(start, end + 1):
        shifted[:, col + length] = image[:, col]
    return shifted


def centerX(digit):
    topLine = getTopLine(digit)
    bottomLine = getBottomLine(digit)
    if topLine is None or bottomLine is None:
        return digit
    centerLine = (topLine + bottomLine) >> 1
    imageCenter = digit.shape[0] >> 1
    digit = rowShift(
        digit, start=topLine, end=bottomLine, length=imageCenter - centerLine)
    return digit


def centerY(digit):
    leftLine = getLeftLine(digit)
    rightLine = getRightLine(digit)
    if leftLine is None or rightLine is None:
        return digit
    centerLine = (leftLine + rightLine) >> 1
    imageCenter = digit.shape[1] >> 1
    digit = colShift(
        digit, start=leftLine, end=rightLine, length=imageCenter - centerLine)
    return digit
