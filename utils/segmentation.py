import cv2
import numpy as np

import utils.steerableFilterALCM as ALCM


class MarkerNotFoundException(Exception):
    pass


def reconnectContours(contours, a, b):
    b = a // 2
    m1 = ALCM.steerableFilterALCM(contours, a, b, 0)
    ret, t1 = cv2.threshold(m1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m2 = ALCM.steerableFilterALCM(contours, a, b, 45)
    ret, t2 = cv2.threshold(m2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m3 = ALCM.steerableFilterALCM(contours, a, b, -45)
    ret, t3 = cv2.threshold(m3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m4 = ALCM.steerableFilterALCM(contours, a, b, 90)
    ret, t4 = cv2.threshold(m4, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return cv2.add(t1, cv2.add(t2, cv2.add(t3, t4)))


def thresholdSegmentation(im, blurSize, ellipseSize, threshSize=25, threshOffset=2):
    blur = im.copy()
    blur = cv2.medianBlur(blur, blurSize)
    blur = cv2.GaussianBlur(blur, (blurSize, blurSize), 0, 0)

    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, threshSize,
                                   threshOffset)

    # contour reconstruction
    a = ellipseSize
    b = a / 2
    ALCM = reconnectContours(thresh, a, b)
    contours, _ = cv2.findContours(ALCM, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    res = np.zeros(im.shape, im.dtype)
    frag = [max(contours, key=cv2.contourArea)]
    cv2.drawContours(res, frag, -1, 255, -1)

    return res


def check_markers_are_found(ids, marker_ids):
    if ids is None:
        return False
    return sum([x in ids for x in marker_ids]) == len(marker_ids)


def loadSegmentationMask(filename):
    ret = cv2.imread(filename)
    ret = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)
    _, ret = cv2.threshold(ret, 0, 255, cv2.THRESH_BINARY)
    return ret


def segmentationDiff(ref, seg):
    h, w = ref.shape[:2]
    res = np.zeros((h, w, 3), np.uint8)

    res[:, :, 0] = cv2.bitwise_and(cv2.bitwise_not(ref), seg)
    res[:, :, 1] = cv2.bitwise_and(ref, seg)
    res[:, :, 2] = cv2.bitwise_and(ref, cv2.bitwise_not(seg))

    refCount = cv2.countNonZero(ref)
    segCount = cv2.countNonZero(seg)
    surplus = cv2.countNonZero(res[:, :, 0])
    common = cv2.countNonZero(res[:, :, 1])
    missing = cv2.countNonZero(res[:, :, 2])

    assert refCount == common + missing
    assert segCount == common + surplus

    return (res, refCount, segCount, common, surplus, missing)


def createMaskVisualization(im, mask, maskOpacity=0.2, imOpacity=0.8):
    if (len(im.shape) == 2 or im.shape[2] == 1):
        im = cv2.cvtColor(im, cv2.GRAY2BGR)
    mask_red = np.zeros(im.shape, im.dtype)
    mask_red[:, :, 2] = mask
    return cv2.addWeighted(im, imOpacity, mask_red, maskOpacity, 0)


def crop_image(image, pixel_value=10, is_lt=False):
    # Remove the zeros padding
    # Find the coordinates of non-zero elements
    if not is_lt:
        indices = np.where(image > pixel_value)
    else:
        indices = np.where(image < pixel_value)

    # Calculate the minimum and maximum row and column indices
    min_row, max_row = min(indices[0]), max(indices[0])
    min_col, max_col = min(indices[1]), max(indices[1])

    # Crop the image to the minimum bounding box
    cropped_image = image[min_row:max_row + 1, min_col:max_col + 1]
    #
    # black_pixels = np.where(
    #     (cropped_image[:, :, 0] == 0) &
    #     (cropped_image[:, :, 1] == 0) &
    #     (cropped_image[:, :, 2] == 0)
    # )
    #
    # # set those pixels to white
    # cropped_image[black_pixels] = [255, 255, 255]

    return cropped_image
