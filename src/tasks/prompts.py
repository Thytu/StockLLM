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

MLM_PROMPT = """<s>[INST] Given an incomplit set of chess moves and the game's score, write the missing chess moves.

Missing chess moves are indicated with a "?" mark. Write ONLY the missing moves, not the provided ones.

Input Format: A comma-separated list of chess moves followed by the game score.
Output Format: A comma-separated list of the missing chess moves.

Moves: {}
Score: {} [/INST]

Output:"""

def generate_MLM_prompt(tokenizer, data_point, return_tensors=None):
    moves, removed_moves = __remove_random_moves(
        moves=data_point["Moves"],
        prob_to_remove=0.025,
    )

    if len(removed_moves) == 0:
        removed_moves = moves[-1]
        moves[-1] = "?"

    full_prompt = MLM_PROMPT.format(moves, data_point["Result"]).replace("'", "")

    result = tokenize(tokenizer, full_prompt, return_tensors=return_tensors)
    result["labels"] = tokenize(tokenizer, str(removed_moves))["input_ids"]

    return result

# def generate_random_prompt(
#     data_sample,
#     tokenizer,
#     probs = {"MLM": 0.13, "MLM_lastonly": 0.40, "regression": 0.47},
# ):
#     value = random()

#     if value <= probs["MLM"]:
#         return generate_MLM_prompt(tokenizer, data_sample)

#     elif value <= probs["MLM"] + probs["MLM_lastonly"]:
#         return generate_MLMlastonly_prompt(tokenizer, data_sample)

#     return generate_regression_prompt(tokenizer, data_sample)
