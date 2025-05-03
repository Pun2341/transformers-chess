from src.transformer import Predictor
from src.tokenizer import process_fen, process_move
import torch
import chess


class Engine:
    def __init__(self, predictor: Predictor, starting_fen: str = None):
        self.predictor = predictor
        if starting_fen:
            self.board = chess.Board(starting_fen)
        else:
            self.board = chess.Board(chess.STARTING_FEN)

    def get_best_move(self):
        legal_moves = self.board.legal_moves
        best_move = None
        best_bucket = -1
        for move in legal_moves:
            bucket = self._get_bucket(
                self.predictor, move.uci()).item()
            print(bucket)
            if bucket > best_bucket:
                best_move = move
        return best_move

    def _get_bucket(self, predictor, move):
        state = process_fen(self.board.fen())
        action = process_move(move)
        sequence = torch.cat([state, action])
        result = predictor.predict(sequence.view(1, 78))
        print(result.shape)
        print(result[0][-1])
        return torch.argmax(result[0][-2])

    def computer_play(self):
        computer_move = self.get_best_move()
        self.board.push(computer_move)
        return computer_move

    def human_play(self, move):
        legal_moves = list(self.board.legal_moves)
        move = self.board.parse_san(move)
        if move in legal_moves:
            self.board.push(move)
            computer_move = self.computer_play()
            return computer_move
        return None
