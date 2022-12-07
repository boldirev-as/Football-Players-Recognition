import json
import re

import cv2
import numpy as np
import pytesseract
from imutils.object_detection import non_max_suppression

pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\vush6\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'


def crop_img_by_polygon(img, polygon):
    # https://stackoverflow.com/questions/48301186/cropping-concave-polygon-from-image-using-opencv-python
    pts = np.array(polygon)
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    croped = img[y: y + h, x: x + w].copy()
    pts = pts - pts.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(croped, croped, mask=mask)
    return dst


def text(img, size, chan):
    scale_percent = int(size)  # Процент от изначального размера
    image = cv2.imread(img)
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)  #
    ret, threshold_image = cv2.threshold(gray, chan, 150, 1, cv2.THRESH_BINARY)
    cv2.imwrite("cropped.png", threshold_image)
    text = pytesseract.image_to_string(threshold_image, config='--psm 11')
    # cv2.imshow("123", threshold_image)
    # cv2.waitKey(0)
    return threshold_image # text


with open("input.json", mode="r") as f:
    data = json.load(f)

# orig = text("4.png", 1000, 125)
text("10.png", 1000, 125)
orig = cv2.imread("10.png")

img = orig.copy()
# for i in range(len(data["main_coords"])):
# coords = list(map(int, data["support_coords"][0].split(",")))
# cropped_img = crop_img_by_polygon(img, np.array([(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]))

# cv2.imwrite("cropped_1.png", cropped_img)

# print(pytesseract.image_to_string(img, config="--psm 11"))

# text1 = text("eric.png", 1000, 125)
# print([x for x in text1.replace("\n", " ").replace("  ", " ").split()
#       if len(x) > 3 and all(map(lambda y: y.isupper(), x))])


net = cv2.dnn.readNet("frozen_east_text_detection.pb")
# img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA)
blob = cv2.dnn.blobFromImage(img, 1.0, (640, 640), (123.68, 116.78, 103.94), True, False)

outputLayers = []
outputLayers.append("feature_fusion/Conv_7/Sigmoid")
outputLayers.append("feature_fusion/concat_3")

net.setInput(blob)
output = net.forward(outputLayers)
scores = output[0]
geometry = output[1]

(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []

# loop over the number of rows
for y in range(0, numRows):
    # extract the scores (probabilities), followed by the geometrical
    # data used to derive potential bounding box coordinates that
    # surround text
    scoresData = scores[0, 0, y]
    xData0 = geometry[0, 0, y]
    xData1 = geometry[0, 1, y]
    xData2 = geometry[0, 2, y]
    xData3 = geometry[0, 3, y]
    anglesData = geometry[0, 4, y]
    # loop over the number of columns
    for x in range(0, numCols):
        # if our score does not have sufficient probability, ignore it
        if scoresData[x] < 0.6:
            continue
        # compute the offset factor as our resulting feature maps will
        # be 4x smaller than the input image
        (offsetX, offsetY) = (x * 4.0, y * 4.0)
        # extract the rotation angle for the prediction and then
        # compute the sin and cosine
        angle = anglesData[x]
        cos = np.cos(angle)
        sin = np.sin(angle)
        # use the geometry volume to derive the width and height of
        # the bounding box
        h = xData0[x] + xData2[x]
        w = xData1[x] + xData3[x]
        # compute both the starting and ending (x, y)-coordinates for
        # the text prediction bounding box
        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
        startX = int(endX - w)
        startY = int(endY - h)
        # add the bounding box coordinates and probability score to
        # our respective lists
        rects.append((startX, startY, endX, endY))
        confidences.append(scoresData[x])

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)

# loop over the bounding boxes
orig = cv2.imread("cropped.png")
rW, rH = orig.shape[1] / 640, orig.shape[0] / 640
for (startX, startY, endX, endY) in boxes:
    # scale the bounding box coordinates based on the respective
    # ratios
    if 0.9 <= (endX - startX) / (endY - startY) <= 1.6:
        continue
    # print((endX - startX) / (endY - startY))
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)
    cropped = crop_img_by_polygon(orig, ((startX, startY), (endX, startY),
                                         (endX, endY), (startX, endY)))
    try:
        surname = pytesseract.image_to_string(cropped, config="--psm 8").strip("\n")
        surname = "".join([x for x in surname if re.match(r"[A-Za-z]", x)])
        print(surname)
    except TypeError as e:
        print(e)

    # draw the bounding box on the image
    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
# show the output image
cv2.imwrite("ans.png", orig)
