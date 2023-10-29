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

MLM_PROMPT = """task: Given an incomplit set of chess moves and the game's score, write the missing chess moves.
Missing chess moves are indicated with a "?" mark. Write ONLY the missing moves, not the provided ones.

Input Format: A comma-separated list of chess moves followed by the game score.
Output Format: A comma-separated list of the missing chess moves.

Moves: {}
Score: {}
Result:"""

def generate_MLM_prompt(tokenizer, data_point, return_tensors=None):
    moves, removed_moves = __remove_random_moves(data_point["Moves"])

    full_prompt = MLM_PROMPT.format(moves, data_point["Result"]).replace("'", "")

    result = tokenize(tokenizer, full_prompt, return_tensors=return_tensors)
    result["labels"] = tokenize(tokenizer, str(removed_moves))["input_ids"]

    return result

REGRESSION_PROMPT = """Task: Given a full set of chess moves, announce the final result of the game.

Input Format: A comma-separated list of chess moves.
Output Format: "1-0" if White wins, "0-1" if Black wins, and "1/2-1/2" in case of a draw.

Moves: {}
Result:"""

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
