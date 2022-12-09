import os
import json


def load_jsonl(filename: str):
    json_list = list(open(filename, "r"))
    dataset = []
    d = ""
    for s in json_list:
        d += s
        if s[0] == "}":
            dataset.append(json.loads(d))
            d = ""
    return dataset


def count_results(results):
    counts = {"simple": 0, "medium": 0, "hard": 0}
    for res in results:
        if res["correct"]:
            counts[res["diffifulty"]] += 1
    counts["all"] = counts["simple"] + counts["medium"] + counts["hard"]
    return counts


def ratio_string(fraction, total):
    return f"{fraction} / {total} ({round(100 * fraction/total, 2)}%)"


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(ROOT_PATH, "results")
for dirname in os.listdir(RESULTS_PATH):
    res_dir = os.path.join(RESULTS_PATH, dirname)

    simple_results = load_jsonl(os.path.join(res_dir, "results-simple.jsonl"))
    medium_results = load_jsonl(os.path.join(res_dir, "results-medium.jsonl"))
    hard_results = load_jsonl(os.path.join(res_dir, "results-hard.jsonl"))

    simple_counts = count_results(simple_results)
    medium_counts = count_results(medium_results)
    hard_counts = count_results(hard_results)

    print(f"\n----- {dirname} Results -----")
    print("Prompts\t\tSimple Acc.\t\t\tMedium Acc.\t\t\tHard Acc.\t\t\tAll Acc.")
    print("-"*140)
    print(
        f"simple\t\t{ratio_string(simple_counts['simple'], 300)}\t\t{ratio_string(simple_counts['medium'], 300)}\t\t{ratio_string(simple_counts['hard'], 300)}\t\t{ratio_string(simple_counts['all'], 900)}")
    print(
        f"medium\t\t{ratio_string(medium_counts['simple'], 300)}\t\t{ratio_string(medium_counts['medium'], 300)}\t\t{ratio_string(medium_counts['hard'], 300)}\t\t{ratio_string(medium_counts['all'], 900)}")
    print(
        f"hard\t\t{ratio_string(hard_counts['simple'], 300)}\t\t{ratio_string(hard_counts['medium'], 300)}\t\t{ratio_string(hard_counts['hard'], 300)}\t\t{ratio_string(hard_counts['all'], 900)}")
    print(
        f"total\t\t{ratio_string(simple_counts['simple'] + medium_counts['simple'] + hard_counts['simple'], 900)}\t\t{ratio_string(simple_counts['medium'] + medium_counts['medium'] + hard_counts['medium'], 900)}\t\t{ratio_string(simple_counts['hard'] + medium_counts['hard'] + hard_counts['hard'], 900)}")
