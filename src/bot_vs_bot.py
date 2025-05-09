import chess
from src.engine import Engine, EngineWithHeuristics
from src.transformer import TransformerConfig, PositionalEncodings, TransformerDecoder, Predictor
from src.utils import MOVE_TO_ACTION
from src.tokenizer import SEQUENCE_LENGTH
import torch


def get_predictor(model_path, transformer_config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerDecoder(transformer_config)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    predictor = Predictor(model)
    return predictor


def play_game(predictor_white, predictor_black, max_moves=250):
    board = chess.Board()
    engine_white = Engine(predictor_white, starting_fen=board.fen())
    # engine_black = Engine(predictor_black, starting_fen=board.fen())
    # engine_white = EngineWithHeuristics(predictor_white, starting_fen=board.fen())
    engine_black = EngineWithHeuristics(predictor_black, starting_fen=board.fen())

    move_count = 0
    while not board.is_game_over() and move_count < max_moves:
        current_engine = engine_white if board.turn == chess.WHITE else engine_black
        move = current_engine.get_best_move()

        if move is None or move not in board.legal_moves:
            break

        board.push(move)

        # Keep engines in sync
        engine_white.board = board.copy()
        engine_black.board = board.copy()

        move_count += 1

    if move_count >= max_moves and not board.is_game_over():
        return "1/2-1/2", move_count, board.fen()  # Treated as draw
    return board.result(), move_count, board.fen()


def main():
    model_path_1 = "src/checkpoint_epoch4_20250506_053148.pt"
    model_path_2 = "src/checkpoint_epoch4_20250506_053148.pt"

    config1 = TransformerConfig(
        vocab_size=len(MOVE_TO_ACTION),
        output_size=128,
        pos_encodings=PositionalEncodings.SINUSOID,
        max_sequence_length=SEQUENCE_LENGTH + 2,
        num_heads=4,
        num_layers=2,
        embedding_dim=64,
        apply_post_ln=True,
        apply_qk_layernorm=False,
        use_causal_mask=False,
    )
    
    config2 = TransformerConfig(
        vocab_size=len(MOVE_TO_ACTION),
        output_size=128,
        pos_encodings=PositionalEncodings.SINUSOID,
        max_sequence_length=SEQUENCE_LENGTH + 2,
        num_heads=4,
        num_layers=2,
        embedding_dim=64,
        apply_post_ln=True,
        apply_qk_layernorm=False,
        use_causal_mask=False,
    )

    predictor1 = get_predictor(model_path_1, config1)
    predictor2 = get_predictor(model_path_2, config2)

    results = {"1_win": 0, "2_win": 0, "draw": 0}
    side_results = {"white_win": 0, "black_win": 0, "draw": 0}

    for i in range(5):
        # Game 1: engine1 as White
        result, moves, fen = play_game(predictor1, predictor2)
        print(f"Game {2*i+1}: {result} ({moves} moves)")
        if result == "1-0":
            results["1_win"] += 1
            side_results["white_win"] += 1
            print("engine 1 won")
        elif result == "0-1":
            results["2_win"] += 1
            side_results["black_win"] += 1
            print("engine 2 won")
        else:
            results["draw"] += 1
            side_results["draw"] += 1

        # Game 2: engine2 as White
        result, moves, fen = play_game(predictor2, predictor1)
        print(f"Game {2*i+2}: {result} ({moves} moves)")
        if result == "1-0":
            results["2_win"] += 1
            side_results["white_win"] += 1
            print("engine 2 won")
        elif result == "0-1":
            results["1_win"] += 1
            side_results["black_win"] += 1
            print("engine 1 won")
        else:
            results["draw"] += 1
            side_results["draw"] += 1

    print("\n=== Summary ===")
    print(f"Model 1 Wins: {results['1_win']}")
    print(f"Model 2 Wins: {results['2_win']}")
    print(f"Draws: {results['draw']}")
    print("\n--- By Color ---")
    print(f"White Wins: {side_results['white_win']}")
    print(f"Black Wins: {side_results['black_win']}")
    print(f"Draws: {side_results['draw']}")

if __name__ == "__main__":
    main()
