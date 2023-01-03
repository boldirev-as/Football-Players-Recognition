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
    counter = 0
    prob_best = ""
    for other_surname in other_cand:
        if other_surname.split()[-1].lower() in [first_name.lower(), second_name.lower()]:
            counter += 1
            prob_best = other_surname

    if counter == 1:
        min_cer = 0.0
        best_surname = prob_best

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
    probe_surnames_scores = []
    for j, prob_surname in enumerate(candidates):
        if len(prob_surname) < 2 or prob_surname.isdigit():
            continue
        min_cer = 1000
        best_surname = ""
        text_part = ""
        for surname in surnames:
            # print(surname.split() + surname.split()[-1].split('-'))
            for i, part in enumerate(surname.split() + surname.split()[-1].split('-')):
                if len(part) < 2:
                    continue
                cer = min(fastwer.score_sent(prob_surname.lower(), part.lower(), char_level=True),
                          fastwer.score_sent(part.lower(), prob_surname.lower(), char_level=True))

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

        if len(other_cand) in [1, 0]:
            probe_surnames_scores.append([best_surname, min_cer])
            continue

        if j - 1 >= 0 and len(candidates) > j + 1:
            print(other_cand, candidates[j - 1] + " " + candidates[j] + " " + candidates[j + 1], min_cer)

        if len(candidates) > j + 1 and min_cer < 50:
            min_cer = 1000
            best_surname, min_cer = choose_best_surname(other_cand, candidates[j],
                                                        candidates[j + 1], best_surname, min_cer)
        if j - 1 >= 0 and min_cer < 50:
            min_cer = 1000
            best_surname, min_cer = choose_best_surname(other_cand, candidates[j - 1],
                                                        candidates[j], best_surname, min_cer)

        if min_cer > 20 and j - 1 >= 0 and candidates[j - 1].isdigit():
            for num, other_surname in zip(other_cand_nums, other_cand):
                if num == candidates[j - 1]:
                    min_cer = 0.1
                    best_surname = other_surname
        if min_cer > 20 and j - 2 >= 0 and candidates[j - 2].isdigit():
            for num, other_surname in zip(other_cand_nums, other_cand):
                if num == candidates[j - 2]:
                    min_cer = 0.1
                    best_surname = other_surname

        probe_surnames_scores.append([best_surname, min_cer])

    probe_surnames_scores.sort(key=lambda x: x[1])
    return probe_surnames_scores


@app.route('/api/image', methods=['POST'])
def upload():
    t = time.time()
    input = {}
    for item in request.form:
        input[item] = request.form[item].replace("\\'", "")
        if item == "team":
            print(input[item])
            input[item] = json.loads(input[item])

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

    candidates = [x.strip(" ") for x in re.sub(r"[^A-Za-z0-9-' ]", " ", text).split()
                  if sum(map(lambda y: y.isalpha(), x)) > 0 or x.isdigit()]
    print(candidates)
    surnames = choose_the_most_suitable_candidates(candidates, team_members, team_numbers)
    print(surnames)

    all_unique_surnames = set()
    all_unique_list = []
    for surname, score in surnames:
        tmp = len(all_unique_surnames)
        all_unique_surnames.add(surname)
        if len(all_unique_surnames) != tmp:
            all_unique_list.append([surname, score])

    main_surnames = [x for x in all_unique_list][:11]
    for i in range(len(main_surnames)):
        if main_surnames[i][0] in change_players:
            main_surnames[i][0] = change_players[main_surnames[i][0]]

    ans_surnames = {"recognized_players": []}
    for surname in main_surnames:
        for member in input["team"]:
            if member["full_name"] == surname[0]:
                ans_surnames["recognized_players"].append(f"{member['id']}, {member['full_name']} "
                                                          f"({100 if surname[1] >= 40 else surname[1]})")
    print("SCORED DURING", time.time() - t)
    return json.dumps(ans_surnames)


if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=3000)
    # app.run(debug=True)
