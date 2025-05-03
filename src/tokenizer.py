import torch
from apache_beam import coders
from utils import MOVE_TO_ACTION, ACTION_TO_MOVE

CODERS = {
    'fen': coders.StrUtf8Coder(),
    'move': coders.StrUtf8Coder(),
    'count': coders.BigIntegerCoder(),
    'win_prob': coders.FloatCoder(),
}

CODERS['action_value'] = coders.TupleCoder((
    CODERS['fen'],
    CODERS['move'],
    CODERS['win_prob'],
))

CODERS['state_value'] = coders.TupleCoder((
    CODERS['fen'],
    CODERS['win_prob'],
))

CODERS['behavioral_cloning'] = coders.TupleCoder((
    CODERS['fen'],
    CODERS['move'],
))


def process_fen(fen: str) -> torch.tensor:
    return _tokenize(fen)


def process_move(move: str) -> torch.tensor:
    return torch.tensor([MOVE_TO_ACTION[move]])


_CHARACTERS = [
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    'a',
    'b',
    'c',
    'd',
    'e',
    'f',
    'g',
    'h',
    'p',
    'n',
    'r',
    'k',
    'q',
    'P',
    'B',
    'N',
    'R',
    'Q',
    'K',
    'w',
    'black',  # Added colors to disambiguate b column from black
    '.',
]

_INDEX = {letter: index for index, letter in enumerate(_CHARACTERS)}
_SPACES = frozenset({'1', '2', '3', '4', '5', '6', '7', '8'})

SEQUENCE_LENGTH = 77


def _tokenize(fen: str) -> torch.tensor:
    """
    Returns an array of tokens from FEN string

    Args:
        fen (str): FEN string to tokenize
    """

    board, color, castling, en_passant, halfmoves, fullmoves = fen.split(' ')
    board = board.replace('/', '')

    indices = [_INDEX['w'] if color == 'w' else _INDEX['black']]

    # Board
    for char in board:
        if char in _SPACES:
            indices.extend([_INDEX['.']] * int(char))
        else:
            indices.append(_INDEX[char])

    # Castling
    if castling == '-':
        indices.extend([_INDEX['.']] * 4)
    else:
        for char in castling:
            indices.append(_INDEX[char])
        if len(castling) < 4:
            indices.extend([_INDEX['.']] * (4 - len(castling)))

    # En passant
    if en_passant == '-':
        indices.extend([_INDEX['.']] * 2)
    else:
        for char in en_passant:
            indices.append(_INDEX[char])

    # Halfmoves
    halfmoves += '.' * (3 - len(halfmoves))
    indices.extend([_INDEX[x] for x in halfmoves])

    # Fullmoves
    fullmoves += '.' * (3 - len(fullmoves))
    indices.extend([_INDEX[x] for x in fullmoves])

    assert len(indices) == SEQUENCE_LENGTH

    return torch.tensor(indices, dtype=torch.uint8)
