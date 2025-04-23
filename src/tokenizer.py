import torch

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


def tokenize(fen: str) -> torch.tensor:
    """Returns an array of tokens from FEN string

    Args:
        fen (str): FEN string to tokenize
    """

    board, color, castling, en_passant, halfmoves, fullmoves = fen.split(' ')
    board = board.replace('/', '')

    indices = [_INDEX['w'] if color == 'w' else _INDEX['black']]

    for char in board:
        if char in _SPACES:
            indices.extend([_INDEX['.']] * int(char))
        else:
            indices.append(_INDEX[char])

    if castling == '-':
        indices.extend([_INDEX['.']] * 4)
    else:
        for char in castling:
            indices.append(_INDEX[char])
        if len(castling) < 4:
            indices.extend([_INDEX['.']] * (4 - len(castling)))

    if en_passant == '-':
        indices.extend([_INDEX['.']] * 2)
    else:
        for char in en_passant:
            indices.append(_INDEX[char])

    halfmoves += '.' * (3 - len(halfmoves))
    indices.extend([_INDEX[x] for x in halfmoves])

    fullmoves += '.' * (3 - len(fullmoves))
    indices.extend([_INDEX[x] for x in fullmoves])

    assert len(indices) == SEQUENCE_LENGTH

    return torch.tensor(indices, dtype=torch.uint8)
