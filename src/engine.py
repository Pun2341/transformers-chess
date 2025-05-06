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
            # print(bucket)
            if bucket > best_bucket:
                best_move = move
                best_bucket = bucket
        return best_move

    def get_best_move_from_fen(self, fen):
        board = chess.Board(fen)
        legal_moves = board.legal_moves
        best_move = None
        best_bucket = -1
        for move in legal_moves:
            bucket = self._get_bucket(
                self.predictor, move).item()
            if bucket > best_bucket:
                best_move = move
                best_bucket = bucket
        return best_move

    def _get_bucket(self, predictor, move):
        state = process_fen(self.board.fen())
        action = process_move(move)
        sequence = torch.cat(
            [state, action, torch.Tensor([0]).to(torch.uint8)])
        result = predictor.predict(sequence.view(1, 79))
        # print(result)
        return torch.argmax(result[0][-1])

    def computer_play(self):
        computer_move = self.get_best_move()
        move_san = self.board.san(computer_move)
        self.board.push(computer_move)
        print("Computer plays:", move_san)
        return computer_move

    def human_play(self, move):
        legal_moves = list(self.board.legal_moves)
        move = self.board.parse_san(move)
        if move in legal_moves:
            self.board.push(move)
            return move
        return None
