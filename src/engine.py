from src.transformer import Predictor
from src.tokenizer import process_fen, process_move
import torch
import chess
import random

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
}

class Engine:
    def __init__(self, predictor: Predictor, starting_fen: str = None):
        self.predictor = predictor
        self.board = chess.Board(starting_fen) if starting_fen else chess.Board(chess.STARTING_FEN)

    def get_best_move(self):
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return None
        best_score = float('-inf')
        best_moves = []

        for move in legal_moves:
            bucket = self._get_bucket(self.predictor, move.uci()).item()
            if bucket > best_score:
                best_moves = [move]
                best_score = bucket
            elif bucket == best_score:
                best_moves.append(move)

        return random.choice(best_moves)

    def get_best_move_from_fen(self, fen):
        self.board = chess.Board(fen)
        legal_moves = self.board.legal_moves
        best_move = []
        best_bucket = -1
        for move in legal_moves:
            bucket = self._get_bucket(
                self.predictor, move.uci()).item()
            if bucket > best_bucket:
                best_move = [move]
                best_bucket = bucket
            elif bucket == best_bucket:
                best_move.append(move)
        rand = random.randint(0, len(best_move)-1)
        print(len(best_move), best_bucket)
        return best_move[rand]

    def _get_bucket(self, predictor, move):
        state = process_fen(self.board.fen())
        action = process_move(move)
        sequence = torch.cat(
            [state, action, torch.Tensor([0]).to(torch.uint8)])
        result = predictor.predict(sequence.view(1, 79))
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


class EngineWithHeuristics(Engine):
    def __init__(self, predictor: Predictor, starting_fen: str = None, weights=None):
        super().__init__(predictor, starting_fen)
        self.weights = weights or {
            'material':     3.0, # 2.8,
            'center':       3.0, # 2.5,
            'check':        10.0, # 3.4, 
            'capture':      15.0, # 2.2, 
            'development':  3.0, # 2.3,
            'blunder':      15.0, # 2.4, 
            'trade':        10.0, # 2.1, 
            'hanging':      15.0,
        }

    def get_best_move(self):
        legal_moves = list(self.board.legal_moves)
        best_moves = []
        best_score = float('-inf')

        for move in legal_moves:
            model_score = self._model_score(move)
            heuristic_score = self._heuristic_score(move)
            total_score = model_score + heuristic_score

            if total_score > best_score:
                best_score = total_score
                best_moves = [move]
            elif total_score == best_score:
                best_moves.append(move)

        return random.choice(best_moves)

    def _model_score(self, move):
        return self._get_bucket(self.predictor, move.uci()).item()

    def _heuristic_score(self, move):
        captured_piece = self.board.piece_at(move.to_square)
        attacker_piece = self.board.piece_at(move.from_square)
        piece = attacker_piece
        development_score = self._weighted_development_score(move, piece)
        prev_color = self.board.turn
        self.board.push(move)

        score = (
            self._weighted_material_score() +
            self._weighted_center_score(move) +
            self._weighted_check_score() +
            self._weighted_capture_score(captured_piece) +
            development_score -
            self._weighted_blunder_penalty(move) + 
            self._weighted_trade_score(move, attacker_piece, captured_piece) - 
            self._weighted_hanging_penalty(prev_color) 
        )

        self.board.pop()
        return score

    # Individual weighted components
    def _weighted_material_score(self):
        return self.weights['material'] * self._evaluate_material()

    def _weighted_center_score(self, move):
        center_squares = {chess.E4, chess.D4, chess.E5, chess.D5}
        center_score = 1 if move.to_square in center_squares else 0
        return self.weights['center'] * center_score

    def _weighted_check_score(self):
        if self.board.is_checkmate():
            return self.weights['check'] * 100
        elif self.board.is_check():
            return self.weights['check'] * 5
        return 0

    def _weighted_capture_score(self, captured_piece):
        if captured_piece and captured_piece.color == self.board.turn:
            return self.weights['capture'] * PIECE_VALUES.get(captured_piece.piece_type, 0)
        return 0

    def _weighted_development_score(self, move, piece):
        return self.weights['development'] * self._evaluate_development(move, piece)

    def _weighted_blunder_penalty(self, move):
        return self.weights['blunder'] * self._is_blunder(move)
    
    def _weighted_trade_score(self, move, attacker_piece, captured_piece):
        if not captured_piece or not attacker_piece:
            return 0

        attacker_square = move.to_square
        can_be_captured = self.board.is_attacked_by(self.board.turn, attacker_square)
        
        victim_value = PIECE_VALUES.get(captured_piece.piece_type, 0)
        attacker_value = PIECE_VALUES.get(attacker_piece.piece_type, 0) if can_be_captured else 0
        trade_value = victim_value - attacker_value

        return self.weights['trade'] * trade_value
    
    def _weighted_hanging_penalty(self, prev_color):
        penalty = 0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.color == prev_color:
                is_attacked = self.board.is_attacked_by(not prev_color, square)
                is_defended = self.board.is_attacked_by(prev_color, square)
                value = PIECE_VALUES.get(piece.piece_type, 0)
                # print(chess.square_name(square))
                # print(is_attacked)
                # print(is_defended)
                # print(value)

                if is_attacked:
                    if not is_defended:
                        penalty += value  # fully hanging
                    else:
                        # Penalize if attackers are cheaper than defenders
                        attackers = self.board.attackers(not prev_color, square)
                        defenders = self.board.attackers(prev_color, square)
                        min_attacker_value = min(
                            [PIECE_VALUES.get(self.board.piece_at(sq).piece_type, 10) for sq in attackers],
                            default=10
                        )
                        min_defender_value = min(
                            [PIECE_VALUES.get(self.board.piece_at(sq).piece_type, 10) for sq in defenders],
                            default=10
                        )
                        if min_attacker_value < min_defender_value:
                            penalty += value * 0.5  # penalize bad trade exposure

        return self.weights['hanging'] * penalty

    def _evaluate_material(self):
        score = 0
        for piece_type, value in PIECE_VALUES.items():
            score += len(self.board.pieces(piece_type, not self.board.turn)) * value
            score -= len(self.board.pieces(piece_type, self.board.turn)) * value
        return score

    def _evaluate_development(self, move, piece):
        if not piece or piece.color != self.board.turn:
            return 0
        if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            rank = chess.square_rank(move.from_square)
            if (self.board.turn == chess.WHITE and rank == 0) or \
            (self.board.turn == chess.BLACK and rank == 7):
                return 1
        return 0

    def _is_blunder(self, move):
        to_sq = move.to_square
        is_attacked = self.board.is_attacked_by(self.board.turn, to_sq)
        is_defended = self.board.is_attacked_by(not self.board.turn, to_sq)
        return 1 if is_attacked and not is_defended else 0
    
    def _evaluate_trade(self, move):
        captured_piece = self.board.piece_at(move.to_square)
        attacker_piece = self.board.piece_at(move.from_square)

        if not captured_piece or not attacker_piece:
            return 0

        victim_value = PIECE_VALUES.get(captured_piece.piece_type, 0)
        attacker_value = PIECE_VALUES.get(attacker_piece.piece_type, 0)

        return victim_value - attacker_value
