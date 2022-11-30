import os
import re
import time
import random
import json
from functools import reduce
import openai
from dotenv import load_dotenv
from datasets import load_dataset
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


def prompt_to_str(prev: str, prompt: dict):
    return prev + "Q: " + prompt["question"] + "\nA: " + prompt["answer"].replace("\n", " ") + "\n\n"


def create_prompts(prompts: list, num_prompts: int):
    prompts = random.sample(prompts, num_prompts)
    return reduce(prompt_to_str, prompts, "")


def ans_to_soln(answer: str) -> float:
    splits = answer.split("#### ")
    if len(splits) > 1:
        num = splits[1]
        num = re.sub(r'[^0-9]', '', num)
        if num:
            return float(num)
        else:
            return float("nan")
    return float("nan")


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


def predict(prompts, p, filename, i):
    prompt = prompts + "Q: " + p["question"] + "\n"
    response = completion_with_backoff(
        model="code-davinci-002",
        prompt=prompt,
        temperature=0,
        max_tokens=256)

    prediction = response["choices"][0]["text"].split("\n\nQ: ")[0]
    res = p.copy()
    res["pred"] = prediction
    res["ans_num"] = ans_to_soln(p["answer"])
    res["pred_num"] = ans_to_soln(prediction)
    res["correct"] = abs(res["ans_num"] - res["pred_num"]) <= 0.1

    with open(filename, "a") as myfile:
        myfile.write(json.dumps(res, indent=4) + "\n")

    return res["correct"]


if __name__ == "__main__":
    SEED = 0
    NUM_PROMPTS = 6

    load_dotenv()
    random.seed(SEED)
    dataset = load_dataset("gsm8k", "main")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # simple:   4805 samples
    # medium:   3593 samples
    # hard:     394 samples
    # total:    8792 samples
    simple, medium, hard = [], [], []
    for split in dataset:
        for d in dataset[split]:
            steps = d["answer"].count("\n")
            if steps <= 3:
                d["diffifulty"] = "simple"
                simple.append(d)
            elif steps <= 6:
                d["diffifulty"] = "medium"
                medium.append(d)
            else:
                d["diffifulty"] = "hard"
                hard.append(d)
    simple = simple[:300]
    medium = medium[:300]
    hard = hard[:300]
    total = simple + medium + hard

    simple_prompts = create_prompts(simple, NUM_PROMPTS)
    medium_prompts = create_prompts(medium, NUM_PROMPTS)
    hard_prompts = create_prompts(hard, NUM_PROMPTS)

    num_correct_simple = 0
    num_correct_medium = 0
    num_correct_hard = 0
    total_prompts = 0
    for i, p in enumerate(total):
        start = time.time()
        simple_correct = predict(
            simple_prompts, p, "results/results-simple.jsonl", i)
        medium_correct = predict(
            medium_prompts, p, "results/results-medium.jsonl", i)
        hard_correct = predict(
            hard_prompts, p, "results/results-hard.jsonl", i)
        end = time.time()

        total_prompts += 1
        if simple_correct:
            num_correct_simple += 1
        if medium_correct:
            num_correct_medium += 1
        if hard_correct:
            num_correct_hard += 1

        print("Prompt #" + str(i) +
              f"\tSimple: {simple_correct}" +
              f"\tSimple Accuracy: {num_correct_simple}/{total_prompts} ({round(100 * num_correct_simple/total_prompts, 2)}%)" +
              f"\tMedium: {medium_correct}" +
              f"\tMedium Accuracy: {num_correct_medium}/{total_prompts} ({round(100 * num_correct_medium/total_prompts, 2)}%)" +
              f"\tHard: {hard_correct}" +
              f"\tHard Accuracy: {num_correct_hard}/{total_prompts} ({round(100 * num_correct_hard/total_prompts, 2)}%)" +
              f"\tTime: {round(end - start, 2)}")
