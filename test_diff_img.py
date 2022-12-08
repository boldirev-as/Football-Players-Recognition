import json
import os
import re
import time

import cv2
import numpy as np
import pytesseract
from imutils.object_detection import non_max_suppression

from flask import Flask, request
from flask_restful import Api
import json
import fastwer

app = Flask(__name__)
api = Api(app)

net = cv2.dnn.readNet("frozen_east_text_detection.pb")
blank = cv2.imread('blank.png')
blank = cv2.resize(blank, (25, 50), interpolation=cv2.INTER_AREA)


def my_dist(a, b):
    n, m = len(a), len(b)
    if n > m:
        # убедимся что n <= m, чтобы использовать минимум памяти O(min(n, m))
        a, b = b, a
        n, m = m, n

    current_row = range(n + 1)  # 0 ряд - просто восходящая последовательность (одни вставки)
    for i in range(1, m + 1):
        previous_row, current_row = current_row, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            current_row[j] = min(add, delete, change)

    return current_row[n]


pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\vush6\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'


def crop_img_by_polygon(img, polygon):
    pts = np.array(polygon)
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    croped = img[y: y + h, x: x + w].copy()
    pts = pts - pts.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(croped, croped, mask=mask)
    return dst


def improve_image(image, size=100, chan=125):
    scale_percent = int(size)
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    ret, threshold_image = cv2.threshold(gray, chan, 150, 1, cv2.THRESH_BINARY)
    cv2.imwrite("cropped.png", threshold_image)
    return cv2.imread("cropped.png")


def get_str_from_img_with_detector(img):
    t = time.time()
    orig = improve_image(img)
    print("IMPROVED", time.time() - t)

    CONV_X, CONV_Y = 960, 960

    t = time.time()
    blob = cv2.dnn.blobFromImage(img, 1.0, (CONV_X, CONV_Y), (123.68, 116.78, 103.94), True, False)
    print("BLOB", time.time() - t)

    outputLayers = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    t = time.time()
    net.setInput(blob)
    output = net.forward(outputLayers)
    scores = output[0]
    geometry = output[1]
    print("FORWARD", time.time() - t)

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        for x in range(0, numCols):
            if scoresData[x] < 0.6:
                continue
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    t = time.time()
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    rW, rH = orig.shape[1] / CONV_X, orig.shape[0] / CONV_Y
    t_sum2 = 0
    t_sum = 0
    whole_text = blank
    for (startX, startY, endX, endY) in boxes:
        # if 0.9 <= (endX - startX) / (endY - startY) <= 1.6:
        #    continue
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH) + 25

        cropped = crop_img_by_polygon(orig, ((startX, startY), (endX, startY),
                                             (endX, endY), (startX, endY)))
        cropped2 = crop_img_by_polygon(img, ((startX, startY), (endX, startY),
                                             (endX, endY), (startX, endY)))
        try:
            # t2 = time.time()
            # cropped = cv2.resize(cropped, (100, 50), interpolation=cv2.INTER_AREA)
            # surname = pytesseract.image_to_string(cropped, config="--psm 8").strip("\n")
            # surname = re.sub(r"[^A-Za-z]", "", surname)
            # output.append(surname)
            # t_sum2 += time.time() - t2
            # cv2.imwrite(f"bin/{surname}.png", cropped)

            # surname = pytesseract.image_to_string(cropped2, config="--psm 8").strip("\n")
            # surname = re.sub(r"[^A-Za-z]", "", surname)
            # output.append(surname)

            t2 = time.time()
            whole_text = np.concatenate((whole_text,
                                         blank,
                                         cv2.resize(cropped, (cropped.shape[0], 50), interpolation=cv2.INTER_AREA),
                                         blank,
                                         cv2.resize(cropped2, (cropped2.shape[0], 50), interpolation=cv2.INTER_AREA)),
                                        axis=1)
            t_sum += time.time() - t2
        except TypeError as e:
            print(e)
            continue
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
        # show the output image
    cv2.imwrite("ans.png", img)
    # cv2.imwrite("whole_text.png", whole_text)
    t2 = time.time()
    surname = pytesseract.image_to_string(whole_text, config="--psm 7").strip("\n")
    t_sum += time.time() - t2
    # print(re.sub(r"[^A-Za-z ]", "", surname).split())
    # print(output)
    print("BOX TO STR", time.time() - t, "CROPPED", t_sum2, "NEW", t_sum)
    output = re.sub(r"[^A-Za-z ]", "", surname).split()
    # cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
    # show the output image
    # cv2.imwrite("ans.png", img)
    return [x for x in output if len(x) > 2]


def get_surnames_from_img(img):
    text = pytesseract.image_to_string(img, config="--psm 11")
    probe_surnames = re.sub(r"[^A-Za-z-]", " ", text).split()

    probe_surnames = [surname for surname in probe_surnames if len(surname) >= 3]

    return probe_surnames


# print(pytesseract.image_to_string(cv2.imread("images/example.png"), config="--psm 11"))
# print(get_str_from_img_with_detector(cv2.imread("images/16.png")))


def form_surnames(candidates):
    candidates = ["".join([x for x in surname if re.match(r"[A-Za-z]", x)]) for surname in candidates]
    candidates = [surname.lower() for surname in candidates if len(surname) >= 3]
    candidates = list(set(candidates))
    return candidates


def get_all_possible_surnames(img, main=False, surnames=None):
    candidates = []

    scale_percent = 400
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    t = time.time()
    candidates.extend(get_surnames_from_img(img))
    print("1 cand", time.time() - t)
    if max([x[1] for x in choose_the_most_suitable_candidates(candidates, surnames)[:(11 if main else 9)]]) <= 30:
        candidates = form_surnames(candidates)
        print(candidates)
        return candidates

    t = time.time()
    candidates.extend(get_surnames_from_img(improve_image(img)))
    print("2 cand", time.time() - t)
    if max([x[1] for x in choose_the_most_suitable_candidates(candidates, surnames)[:(11 if main else 9)]]) <= 30:
        candidates = form_surnames(candidates)
        print(candidates)
        return candidates

    if main:
        t = time.time()
        candidates.extend((get_str_from_img_with_detector(img)))
        print("3 cand", time.time() - t)

    candidates = ["".join([x for x in surname if re.match(r"[A-Za-z]", x)]) for surname in candidates]
    candidates = [surname.lower() for surname in candidates if len(surname) >= 3]
    candidates = list(set(candidates))
    print(candidates)
    return candidates


def choose_the_most_suitable_candidates(candidates, surnames):
    t = time.time()
    probe_surnames_scores = []
    for prob_surname in candidates:
        min_cer = 1000
        best_surname = ""
        for surname in surnames:
            if len(surname) == 0:
                continue
            cer = fastwer.score_sent(prob_surname.lower(), surname.lower().split()[-1], char_level=True)
            if cer < min_cer:
                min_cer = cer
                best_surname = surname
            if cer == 0:
                break
        probe_surnames_scores.append([best_surname, min_cer])

    probe_surnames_scores.sort(key=lambda x: x[1])
    print("SCORED DURING", time.time() - t)
    return probe_surnames_scores


def answer_for_img(img, coords, surnames, num_players, main=False):
    cropped_img = crop_img_by_polygon(img, np.array([(coords[i], coords[i + 1])
                                                     for i in range(0, len(coords), 2)]))
    main_players = choose_the_most_suitable_candidates(get_all_possible_surnames(cropped_img, main, surnames), surnames)

    top_players = set()
    scores = []
    i = 0
    while len(top_players) < num_players and i < len(main_players):
        previous = len(top_players)
        top_players.add(main_players[i][0])
        if previous != len(top_players):
            scores.append(main_players[i][1])
        i += 1
    return scores, top_players


@app.route('/api/image', methods=['POST'])
def upload():
    t = time.time()

    original_image = request.files['image']
    original_image.save("input.png")

    input = {}
    for item in request.form:
        input[item] = request.form[item].replace("\\'", "")
        if item == "team":
            input[item] = json.loads(input[item])

    original_img = cv2.imread("input.png")
    team_members = [player["full_name"] for player in input["team"]]

    main_players = []
    support_players = []
    scores_main = []
    scores_support = []
    for state in [("coords_lineup_1", "coords_subs_1"), ("coords_lineup_2", "coords_subs_2"),
                  ("coords_lineup_3", "coords_subs_3")]:
        if input[state[0]] == "":
            continue
        coords = list(map(int, input[state[0]].split(",")))
        scores_main, main_players = answer_for_img(original_img, coords, team_members, 11, True)

        if input[state[1]] == "":
            continue
        coords = list(map(int, input[state[1]].split(",")))
        scores_support, support_players = answer_for_img(original_img, coords, team_members, 9)

    main_recognized = sum([1 for x in scores_main if x <= 50])
    support_recognized = sum([1 for x in scores_support if x <= 50])

    print((time.time() - t))

    return json.dumps({
        "code1": 1 if main_recognized >= 11 else 0,
        "code2": f"{main_recognized}/{support_recognized}",
        "code3": 1 if main_recognized >= 11 else 0,
        "lineups": list(main_players),
        "substitutions": list(support_players)
    })


if __name__ == "__main__":
    # from waitress import serve
    # serve(app, host="0.0.0.0", port=3000)
    app.run(debug=True)
