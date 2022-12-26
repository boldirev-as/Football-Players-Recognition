import re
import time

from unidecode import unidecode

from flask import Flask, request
from flask_restful import Api
import json
import fastwer

app = Flask(__name__)
api = Api(app)


def choose_best_surname(other_cand, first_name, second_name, best_surname, min_cer):
    for other_surname in other_cand:
        cer = fastwer.score_sent(first_name.lower() + " " + second_name.lower(),
                                 other_surname.lower(), char_level=True)
        if other_surname.split()[0][0].lower() == first_name[0].lower():
            cer = 0.1
        if cer < min_cer:
            min_cer = cer
            best_surname = other_surname
    for other_surname in other_cand:
        cer = fastwer.score_sent(second_name.lower() + " " + first_name.lower(),
                                 other_surname.lower(), char_level=True)
        if other_surname.split()[-1][0].lower() == first_name[0].lower() and cer > 0.1:
            cer = 0.3
        if cer < min_cer:
            min_cer = cer
            best_surname = other_surname
    return best_surname, min_cer


def choose_the_most_suitable_candidates(candidates, surnames, team_numbers):
    t = time.time()
    probe_surnames_scores = []
    for j, prob_surname in enumerate(candidates):
        if len(prob_surname) < 3:
            continue
        min_cer = 1000
        best_surname = ""
        text_part = ""
        for surname in surnames:
            for i, part in enumerate(surname.split()):
                if len(part) <= 2:
                    continue
                cer = fastwer.score_sent(prob_surname.lower(), part.lower(), char_level=True)
                if cer < min_cer:
                    min_cer = cer
                    best_surname = surname
                    text_part = part
                if cer == 0:
                    break

        other_cand = []
        other_cand_nums = []
        for k, other_surname in enumerate(surnames):
            if text_part.lower() in other_surname.lower():
                other_cand.append(other_surname)
                other_cand_nums.append(team_numbers[k])

        if len(other_cand) == 1:
            probe_surnames_scores.append([best_surname, min_cer])
            continue

        min_cer = 1000
        # print(other_cand, candidates[j - 1] + " " + candidates[j] + " " + candidates[j + 1])
        if len(candidates) > j + 1:
            best_surname, min_cer = choose_best_surname(other_cand, candidates[j], candidates[j + 1], best_surname, min_cer)
        if j - 1 >= 0:
            best_surname, min_cer = choose_best_surname(other_cand, candidates[j - 1], candidates[j], best_surname, min_cer)

        if min_cer > 20 and candidates[j - 1].isdigit():
            for num, other_surname in zip(other_cand_nums, other_cand):
                if num == candidates[j - 1]:
                    min_cer = 0.1
                    best_surname = other_surname

        probe_surnames_scores.append([best_surname, min_cer])

    probe_surnames_scores.sort(key=lambda x: x[1])
    print("SCORED DURING", time.time() - t)
    return probe_surnames_scores


@app.route('/api/image', methods=['POST'])
def upload():
    input = {}
    for item in request.form:
        input[item] = request.form[item].replace("\\'", "")
        if item == "team":
            input[item] = json.loads(input[item])
            print(input[item])

    team_members = [player["full_name"].replace("(C)", "").replace(" De la Flor", "").replace("(TW)", "") for player in
                    input["team"]]
    extra_members = []
    extra_nums = []
    change_players = {}
    for member in input["team"]:
        for field in ["fio1", "fio2", "fio3"]:
            if member[field] is not None and member[field] != "":
                extra_members.append(member[field])
                change_players[member[field]] = member["full_name"]
                extra_nums.append(member["number"])
    print(change_players)
    team_numbers = [player["number"] for player in input["team"]]
    team_numbers.extend(extra_nums)
    team_members.extend(extra_members)

    text = unidecode(request.form["text"]).replace("(C)", "").replace(".", " ").replace("0", "O")
    for repl in range(10):
        text = text.replace(str(repl) + "-", str(repl) + " ")

    candidates = [x.strip(" ") for x in re.sub(r"[^A-Za-z0-9- ]", " ", text).split() if len(x.strip(" ")) > 0]
    surnames = choose_the_most_suitable_candidates(candidates, team_members, team_numbers)
    print(surnames)

    all_unique_surnames = set()
    all_unique_list = []
    for surname, score in surnames:
        tmp = len(all_unique_surnames)
        all_unique_surnames.add(surname)
        if len(all_unique_surnames) != tmp:
            all_unique_list.append([surname, score])

    main_surnames = [x for x in all_unique_list if x[1] < 80][:11]
    for i in range(len(main_surnames)):
        if main_surnames[i][0] in change_players:
            main_surnames[i][0] = change_players[main_surnames[i][0]]

    ans_surnames = {"recognized_players": []}
    for surname in main_surnames:
        for member in input["team"]:
            if member["full_name"] == surname[0]:
                ans_surnames["recognized_players"].append(f"{member['id']}, {member['full_name']} ({surname[1]})")
    return json.dumps(ans_surnames)


if __name__ == "__main__":
    # from waitress import serve

    # serve(app, host="0.0.0.0", port=3000)
    app.run(debug=True)
