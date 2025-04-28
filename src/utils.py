import chess
from enum import Enum

_CHESS_FILE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


class Policy(Enum):
    BEHAVIORAL_CLONING = 1
    ACTION_VALUE = 2
    STATE_VALUE = 3


def _compute_all_possible_moves() -> tuple[dict[str, int], dict[int, str]]:
    """
    Compute all possible moves in a chess game represented in UCI Notation.

    Move - a string representing a playable move in UCI notation.
    Action - an unique integer representing a move.

    Returns a mapping of each move to action and action to move.
    """

    board = chess.BaseBoard.empty()
    moves = []

    for square in range(64):
        next_squares = []

        # Moves for P, B, R, Q, K
        board.set_piece_at(square, chess.Piece.from_symbol("Q"))
        next_squares += board.attacks(square)

        # Moves for knight
        board.set_piece_at(square, chess.Piece.from_symbol("N"))
        next_squares += board.attacks(square)

        board.remove_piece_at(square)

        for next_square in next_squares:
            moves.append(chess.square_name(square) +
                         chess.square_name(next_square))

    for rank, next_rank in [('2', '1'), ('7', '8')]:
        for index_file, file in enumerate(_CHESS_FILE):
            # Normal promotions.
            move = f'{file}{rank}{file}{next_rank}'
            moves += [(move + piece) for piece in ['q', 'r', 'b', 'n']]

            # Capture promotions.
            # Left side.
            if file > 'a':
                next_file = _CHESS_FILE[index_file - 1]
                move = f'{file}{rank}{next_file}{next_rank}'
                moves += [(move + piece) for piece in ['q', 'r', 'b', 'n']]

            # Right side.
            if file < 'h':
                next_file = _CHESS_FILE[index_file + 1]
                move = f'{file}{rank}{next_file}{next_rank}'
                moves += [(move + piece) for piece in ['q', 'r', 'b', 'n']]

    assert len(moves) == 1968, f"Expected 1968 moves, got {len(moves)}"

    move_to_action = {}
    action_to_move = {}

    for action, move in enumerate(moves):
        assert (move not in move_to_action)
        move_to_action[move] = action
        action_to_move[action] = move

    return move_to_action, action_to_move


MOVE_TO_ACTION, ACTION_TO_MOVE = _compute_all_possible_moves()
