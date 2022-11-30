import os
import json
import random
from functools import reduce
from typing import Callable
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


def create_prompts(prompts: list, num_prompts: int, prompt_to_str: Callable) -> str:
    prompts = random.sample(prompts, num_prompts)
    return reduce(prompt_to_str, prompts, "")


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


def predict(prompts: str, p: dict, filename: str, ans_to_soln: Callable) -> bool:
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

    mode = "a" if os.path.exists(filename) else "w"
    with open(filename, mode) as myfile:
        myfile.write(json.dumps(res, indent=4) + "\n")

    return res["correct"]
