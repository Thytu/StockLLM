from itertools import combinations
from model import get_tokenizer


ROWS = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h')
COLUMNS = ('1', '2', '3', '4', '5', '6', '7', '8')


def get_dict_move_in_utf8_to_token(model_vocab_size: int = 32_000):

    positions_in_utf8 = []
    tokenizer = get_tokenizer()

    for row in ROWS:
        for column in COLUMNS:
            positions_in_utf8.append(row + column)

    moves_in_utf8 = []
    
    for (pos_1, pos_2) in list(combinations(positions_in_utf8, 2)):
        moves_in_utf8 += [pos_1 + pos_2, pos_2 + pos_1]

    tokens_start_at = model_vocab_size - len(moves_in_utf8)

    return {
        utf_8: token_id for (utf_8, token_id) in zip(
            moves_in_utf8,
            [tokenizer.decode(i) for i in range(tokens_start_at, tokens_start_at + len(moves_in_utf8))]
        )
    }
