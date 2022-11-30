import json


def load_jsonl(filename: str):
    json_list = list(open(filename, "r"))
    return [json.loads("".join(json_list[i:i+9])) for i in range(0, len(json_list), 9)]


def count_results(results):
    counts = {"simple": 0, "medium": 0, "hard": 0}
    for res in results:
        if res["correct"]:
            counts[res["diffifulty"]] += 1
    counts["all"] = counts["simple"] + counts["medium"] + counts["hard"]
    return counts


def ratio_string(fraction, total):
    return f"{fraction} / {total} ({round(100 * fraction/total, 2)}%)"


simple_results = load_jsonl("results/results-simple.jsonl")
medium_results = load_jsonl("results/results-medium.jsonl")
hard_results = load_jsonl("results/results-hard.jsonl")

simple_counts = count_results(simple_results)
medium_counts = count_results(medium_results)
hard_counts = count_results(hard_results)

print("Prompts\t\tSimple\t\t\tMedium\t\t\tHard\t\t\tAll")
print("-"*110)
print(
    f"simple\t\t{ratio_string(simple_counts['simple'], 300)}\t{ratio_string(simple_counts['medium'], 300)}\t{ratio_string(simple_counts['hard'], 300)}\t{ratio_string(simple_counts['all'], 900)}")
print(
    f"medium\t\t{ratio_string(medium_counts['simple'], 300)}\t{ratio_string(medium_counts['medium'], 300)}\t{ratio_string(medium_counts['hard'], 300)}\t{ratio_string(medium_counts['all'], 900)}")
print(
    f"hard\t\t{ratio_string(hard_counts['simple'], 300)}\t{ratio_string(hard_counts['medium'], 300)}\t{ratio_string(hard_counts['hard'], 300)}\t{ratio_string(hard_counts['all'], 900)}")
