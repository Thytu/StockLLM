from random import random


def tokenize(tokenizer, prompt, return_tensors=None):
    return tokenizer(
        prompt,
        truncation=True,
        max_length=2048,
        padding="max_length",
        return_tensors=return_tensors,
    )


def __remove_random_moves(moves, prob_to_remove: float = 0.1):
    remaining_moves = []
    removed_moves = []

    for m in moves:
        if random() <= prob_to_remove:
            removed_moves.append(m)
            remaining_moves.append("?")
        else:
            remaining_moves.append(m)

    return remaining_moves, removed_moves

MLM_PROMPT = """task: Given an incomplit set of chess moves, write the missing chess moves.
Missing chess moves are indicated with a "?" mark. Write ONLY the missing moves, not the provided ones.

Input Format: A comma-separated list of chess moves.
Output Format: A comma-separated list of the missing chess moves.

--------------------------------------

Example:

Input: "e2e4, e7e6, d2d4, d7d5, b1c3, ?, e4e5, c7c5, d1g4, g8e7, d4c5, b4c3, b2c3"
Output: "f8b4"

--------------------------------------

Moves: {}
Result:
"""

def generate_MLM_prompt(tokenizer, data_point, return_tensors=None):
    moves, removed_moves = __remove_random_moves(data_point["Moves"])

    full_prompt =MLM_PROMPT.format(moves).replace("'", "")

    result = tokenize(tokenizer, full_prompt, return_tensors=return_tensors)
    result["labels"] = tokenize(tokenizer, str(removed_moves))["input_ids"]

    return result

REGRESSION_PROMPT = """Task: Given a full set of chess moves, announce the final result of the game.

Input Format: A comma-separated list of chess moves.
Output Format: "1-0" if White wins, "0-1" if Black wins, and "1/2-1/2" in case of a draw.

--------------------------------------

Example:
Input: "[previous moves], e5d4, b4b3, a6a1, b3b4, a1b1, b4a3, d4c5, a3a4, b1b4"
Output: "0-1"

--------------------------------------

Moves: {}
Result:
"""

def generate_regression_prompt(tokenizer, data_point, return_tensors=None):
    full_prompt = REGRESSION_PROMPT.format(data_point["Moves"]).replace("'", "")

    result = tokenize(tokenizer, full_prompt, return_tensors=return_tensors)
    result["labels"] = tokenize(tokenizer, data_point["Result"])["input_ids"]

    return result


def generate_random_prompt(
    data_sample,
    tokenizer,
    generate_MLM_prompt_prob: float = 0.8,
):
    if random() <= generate_MLM_prompt_prob:
        return generate_MLM_prompt(tokenizer, data_sample)

    return generate_regression_prompt(tokenizer, data_sample)
